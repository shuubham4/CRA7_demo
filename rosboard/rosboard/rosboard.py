#!/usr/bin/env python3

import asyncio
import importlib
import os
import pathlib
import threading
import time
import string
import tornado, tornado.web, tornado.websocket
import traceback
import ssl
import torch
from torchaudio import load, save
import torchaudio
import soundfile as sf
import numpy as np
from scipy.signal import resample
from std_msgs.msg import String
from transformers import WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, WhisperFeatureExtractor

import argparse
import glob
import warnings
from typing import List, Optional, Tuple, Union
from loguru import logger
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from df.checkpoint import load_model as load_model_cp
from df.config import config
from df.io import load_audio, resample, save_audio
from df.logger import init_logger
from df.model import ModelParams
from df.modules import get_device
from df.utils import as_complex, as_real, download_file, get_cache_dir, get_norm_alpha
from df.version import version
from libdf import DF, erb, erb_norm, unit_norm
PRETRAINED_MODELS = ("DeepFilterNet", "DeepFilterNet2", "DeepFilterNet3")
DEFAULT_MODEL = "DeepFilterNet3"


if os.environ.get("ROS_VERSION") == "1":
    import rospy # ROS1
elif os.environ.get("ROS_VERSION") == "2":
    import rosboard.rospy2 as rospy # ROS2
else:
    print("ROS not detected. Please source your ROS environment\n(e.g. 'source /opt/ros/DISTRO/setup.bash')")
    exit(1)

from rosgraph_msgs.msg import Log

from rosboard.serialization import ros2dict
from rosboard.subscribers.dmesg_subscriber import DMesgSubscriber
from rosboard.subscribers.processes_subscriber import ProcessesSubscriber
from rosboard.subscribers.system_stats_subscriber import SystemStatsSubscriber
from rosboard.subscribers.dummy_subscriber import DummySubscriber
from rosboard.handlers import ROSBoardSocketHandler, NoCacheStaticFileHandler

import warnings
import logging

# Suppress all Python warnings
warnings.filterwarnings("ignore")

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress specific logging messages
logging.getLogger('tensorflow').setLevel(logging.ERROR)


class ROSBoardNode(object):
    instance = None
    def __init__(self, node_name = "rosboard_node"):
        self.__class__.instance = self
        rospy.init_node(node_name)
        self.port = rospy.get_param("~port", 8888)
        self.input_pub = rospy.Publisher('web_input', String, queue_size=10)

        # desired subscriptions of all the websockets connecting to this instance.
        # these remote subs are updated directly by "friend" class ROSBoardSocketHandler.
        # this class will read them and create actual ROS subscribers accordingly.
        # dict of topic_name -> set of sockets
        self.remote_subs = {}

        # actual ROS subscribers.
        # dict of topic_name -> ROS Subscriber
        self.local_subs = {}

        # minimum update interval per topic (throttle rate) amang all subscribers to a particular topic.
        # we can throw data away if it arrives faster than this
        # dict of topic_name -> float (interval in seconds)
        self.update_intervals_by_topic = {}

        # last time data arrived for a particular topic
        # dict of topic_name -> float (time in seconds)
        self.last_data_times_by_topic = {}
        self.user_input = ""
        self.audio_input = False
        
        self.tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-medium", language="English", task="transcribe")
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-medium", language="English", task="transcribe", sampling_rate = 16000)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-medium", device="cuda")
        #model = WhisperForConditionalGeneration.from_pretrained("/home/shuubham/Desktop/spine1_train/whisper_train/whisper_arl_medium/checkpoint-2500", return_dict=False)

        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium", return_dict=False)
        self.model.config.forced_decoder_ids = None 
        self.model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="English", task = "transcribe")
        self.model.config.suppress_tokens = []
        self.model.config.use_cache = False
        self.model.config.condition_on_previous_text = False
        
        if rospy.__name__ == "rospy2":
            # ros2 hack: need to subscribe to at least 1 topic
            # before dynamic subscribing will work later.
            # ros2 docs don't explain why but we need this magic.
            self.sub_rosout = rospy.Subscriber("/rosout", Log, lambda x:x)

        tornado_settings = {
            'debug': True, 
            'static_path': os.path.join(os.path.dirname(os.path.realpath(__file__)), 'html')
        }

        tornado_handlers = [
                (r"/rosboard/v1", ROSBoardSocketHandler, {
                    "node": self,
                }),
                (r"/(.*)", NoCacheStaticFileHandler, {
                    "path": tornado_settings.get("static_path"),
                    "default_filename": "index.html"
                }),
        ]

        certfile_path = os.path.join(pathlib.Path(__file__).resolve().parents[1], "cert.pem")
        keyfile_path = os.path.join(pathlib.Path(__file__).resolve().parents[1], "key.pem")

        ssl_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_ctx.load_cert_chain(certfile=certfile_path, keyfile=keyfile_path)

        self.event_loop = None
        self.tornado_application = tornado.web.Application(tornado_handlers, **tornado_settings)
        asyncio.set_event_loop(asyncio.new_event_loop())
        self.event_loop = tornado.ioloop.IOLoop()
        self.tornado_application.listen(self.port, ssl_options=ssl_ctx)

        # allows tornado to log errors to ROS
        self.logwarn = rospy.logwarn
        self.logerr = rospy.logerr

        # tornado event loop. all the web server and web socket stuff happens here
        threading.Thread(target = self.event_loop.start, daemon = True).start()

        # loop to sync remote (websocket) subs with local (ROS) subs
        threading.Thread(target = self.sync_subs_loop, daemon = True).start()

        # loop to keep track of latencies and clock differences for each socket
        threading.Thread(target = self.pingpong_loop, daemon = True).start()

        threading.Thread(target = self.pub_loop, daemon = True).start()

        threading.Thread(target = self.audio_loop, daemon = True).start()

        self.lock = threading.Lock()

        rospy.loginfo("ROSboard listening on :%d" % self.port)

    def start(self):
        rospy.spin()

    def audio_loop(self):
        # time.sleep(5)
        # self.input_pub.publish('Program started')
        while True:
            time.sleep(1)
            if self.audio_input:
                # time.sleep(1)
                path = os.path.join(pathlib.Path(__file__).resolve().parents[1], "audiooutput.wav")
                transcription = self.process_audio(path)
                self.input_pub.publish(transcription)
                self.audio_input = False

    def process_audio(self, path):
        audio, sample_rate = torchaudio.load(path)
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        audio = resampler(audio)
        # replace audio with enhanced audio here.
        resampled_audio = np.array(audio)
        input_features = self.feature_extractor(resampled_audio, sampling_rate=16000, return_tensors="pt").input_features
        generated_ids = self.model.generate(inputs=input_features,no_repeat_ngram_size=4, language="English")
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        transcription = transcription.translate(str.maketrans('', '', string.punctuation))
        transcription = transcription.lower()

        return transcription[1:]

    # integrating code for DFNet 3 #############

    def main_dfnet(args):

        class AudioDataset(Dataset):
            def __init__(self, files: List[str], sr: int) -> None:
                super().__init__()
                self.files = []
                for file in files:
                    if not os.path.isfile(file):
                        logger.warning(f"File not found: {file}. Skipping...")
                    self.files.append(file)
                self.sr = sr

            def __getitem__(self, index) -> Tuple[str, Tensor, int]:
                fn = self.files[index]
                audio, meta = load_audio(fn, self.sr, "cpu")
                return fn, audio, meta.sample_rate

            def __len__(self):
                return len(self.files)
        
        model, df_state, suffix, epoch = init_df(
            args.model_base_dir,
            post_filter=args.pf,
            log_level=args.log_level,
            config_allow_defaults=True,
            epoch=args.epoch,
            mask_only=args.no_df_stage,
        )
        suffix = suffix if args.suffix else None
        if args.output_dir is None:
            args.output_dir = "."
        elif not os.path.isdir(args.output_dir):
            os.mkdir(args.output_dir)
        df_sr = ModelParams().sr
        if args.noisy_dir is not None:
            if len(args.noisy_audio_files) > 0:
                logger.error("Only one of `noisy_audio_files` or `noisy_dir` arguments are supported.")
                exit(1)
            input_files = glob.glob(args.noisy_dir + "/*")
        else:
            assert len(args.noisy_audio_files) > 0, "No audio files provided"
            input_files = args.noisy_audio_files
        ds = AudioDataset(input_files, df_sr)
        loader = DataLoader(ds, num_workers=2, pin_memory=True)
        n_samples = len(ds)
        for i, (file, audio, audio_sr) in enumerate(loader):
            file = file[0]
            audio = audio.squeeze(0)
            progress = (i + 1) / n_samples * 100
            t0 = time.time()
            audio = enhance(
                model, df_state, audio, pad=args.compensate_delay, atten_lim_db=args.atten_lim
            )
            t1 = time.time()
            t_audio = audio.shape[-1] / df_sr
            t = t1 - t0
            rtf = t / t_audio
            fn = os.path.basename(file)
            p_str = f"{progress:2.0f}% | " if n_samples > 1 else ""
            logger.info(f"{p_str}Enhanced noisy audio file '{fn}' in {t:.2f}s (RT factor: {rtf:.3f})")
            audio = resample(audio.to("cpu"), df_sr, audio_sr)
            save_audio(file, audio, sr=audio_sr, output_dir=args.output_dir, suffix=suffix, log=False)
    
    
    def init_df(
        model_base_dir: Optional[str] = None,
        post_filter: bool = False,
        log_level: str = "INFO",
        log_file: Optional[str] = "enhance.log",
        config_allow_defaults: bool = True,
        epoch: Union[str, int, None] = "best",
        default_model: str = DEFAULT_MODEL,
        mask_only: bool = False,
    ) -> Tuple[nn.Module, DF, str, int]:

        try:
            from icecream import ic, install

            ic.configureOutput(includeContext=True)
            install()
        except ImportError:
            pass
        use_default_model = model_base_dir is None or model_base_dir in PRETRAINED_MODELS
        model_base_dir = default_model

        if not os.path.isdir(model_base_dir):
            raise NotADirectoryError("Base directory not found at {}".format(model_base_dir))
        log_file = os.path.join(model_base_dir, log_file) if log_file is not None else None
        init_logger(file=log_file, level=log_level, model=model_base_dir)
        if use_default_model:
            logger.info(f"Using {default_model} model at {model_base_dir}")
        config.load(
            os.path.join(model_base_dir, "config.ini"),
            config_must_exist=True,
            allow_defaults=config_allow_defaults,
            allow_reload=True,
        )
        if post_filter:
            config.set("mask_pf", True, bool, ModelParams().section)
            try:
                beta = config.get("pf_beta", float, ModelParams().section)
                beta = f"(beta: {beta})"
            except KeyError:
                beta = ""
            logger.info(f"Running with post-filter {beta}")
        p = ModelParams()
        df_state = DF(
            sr=p.sr,
            fft_size=p.fft_size,
            hop_size=p.hop_size,
            nb_bands=p.nb_erb,
            min_nb_erb_freqs=p.min_nb_freqs,
        )
        checkpoint_dir = os.path.join(model_base_dir, "checkpoints")
        load_cp = epoch is not None and not (isinstance(epoch, str) and epoch.lower() == "none")
        if not load_cp:
            checkpoint_dir = None
        mask_only = mask_only or config(
            "mask_only", cast=bool, section="train", default=False, save=False
        )
        model, epoch = load_model_cp(checkpoint_dir, df_state, epoch=epoch, mask_only=mask_only)
        if (epoch is None or epoch == 0) and load_cp:
            logger.error("Could not find a checkpoint")
            exit(1)
        logger.debug(f"Loaded checkpoint from epoch {epoch}")
        model = model.to(get_device())
        # Set suffix to model name
        suffix = os.path.basename(os.path.abspath(model_base_dir))
        if post_filter:
            suffix += "_pf"
        logger.info("Running on device {}".format(get_device()))
        logger.info("Model loaded")
        return model, df_state, suffix, epoch

    def df_features(audio: Tensor, df: DF, nb_df: int, device=None) -> Tuple[Tensor, Tensor, Tensor]:
        spec = df.analysis(audio.numpy())  # [C, Tf] -> [C, Tf, F]
        a = get_norm_alpha(False)
        erb_fb = df.erb_widths()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            erb_feat = torch.as_tensor(erb_norm(erb(spec, erb_fb), a)).unsqueeze(1)
        spec_feat = as_real(torch.as_tensor(unit_norm(spec[..., :nb_df], a)).unsqueeze(1))
        spec = as_real(torch.as_tensor(spec).unsqueeze(1))
        if device is not None:
            spec = spec.to(device)
            erb_feat = erb_feat.to(device)
            spec_feat = spec_feat.to(device)
        return spec, erb_feat, spec_feat


    @torch.no_grad()
    def enhance(
        model: nn.Module, df_state: DF, audio: Tensor, pad=True, atten_lim_db: Optional[float] = None
    ):
        model.eval()
        bs = audio.shape[0]
        if hasattr(model, "reset_h0"):
            model.reset_h0(batch_size=bs, device=get_device())
        orig_len = audio.shape[-1]
        n_fft, hop = 0, 0
        if pad:
            n_fft, hop = df_state.fft_size(), df_state.hop_size()
            # Pad audio to compensate for the delay due to the real-time STFT implementation
            audio = F.pad(audio, (0, n_fft))
        nb_df = getattr(model, "nb_df", getattr(model, "df_bins", ModelParams().nb_df))
        spec, erb_feat, spec_feat = df_features(audio, df_state, nb_df, device=get_device())
        enhanced = model(spec.clone(), erb_feat, spec_feat)[0].cpu()
        enhanced = as_complex(enhanced.squeeze(1))
        if atten_lim_db is not None and abs(atten_lim_db) > 0:
            lim = 10 ** (-abs(atten_lim_db) / 20)
            enhanced = as_complex(spec.squeeze(1).cpu()) * lim + enhanced * (1 - lim)
        audio = torch.as_tensor(df_state.synthesis(enhanced.numpy()))
        if pad:
            assert n_fft % hop == 0  # This is only tested for 50% and 75% overlap
            d = n_fft - hop
            audio = audio[:, d : orig_len + d]
        return audio


    def setup_df_argument_parser(
        default_log_level: str = "INFO", parser=None
    ) -> argparse.ArgumentParser:
        if parser is None:
            parser = argparse.ArgumentParser()
        parser.add_argument(
            "--model-base-dir",
            "-m",
            type=str,
            default=None,
            help="Model directory containing checkpoints and config. "
            "To load a pretrained model, you may just provide the model name, e.g. `DeepFilterNet`. "
            "By default, the pretrained DeepFilterNet2 model is loaded.",
        )
        parser.add_argument(
            "--pf",
            help="Post-filter that slightly over-attenuates very noisy sections.",
            action="store_true",
        )
        parser.add_argument(
            "--output-dir",
            "-o",
            type=str,
            default=None,
            help="Directory in which the enhanced audio files will be stored.",
        )
        parser.add_argument(
            "--log-level",
            type=str,
            default=default_log_level,
            help="Logger verbosity. Can be one of (debug, info, error, none)",
        )
        parser.add_argument("--debug", "-d", action="store_const", const="DEBUG", dest="log_level")
        parser.add_argument(
            "--epoch",
            "-e",
            default="best",
            type=parse_epoch_type,
            help="Epoch for checkpoint loading. Can be one of ['best', 'latest', <int>].",
        )
        #parser.add_argument("-v", "--version", action=PrintVersion)
        return parser


    def run():
        parser = setup_df_argument_parser()
        parser.add_argument(
            "--no-delay-compensation",
            dest="compensate_delay",
            action="store_false",
            help="Dont't add some paddig to compensate the delay introduced by the real-time STFT/ISTFT implementation.",
        )
        parser.add_argument(
            "--atten-lim",
            "-a",
            type=int,
            default=None,
            help="Attenuation limit in dB by mixing the enhanced signal with the noisy signal.",
        )
        parser.add_argument(
            "noisy_audio_files",
            type=str,
            nargs="*",
            help="List of noise files to mix with the clean speech file.",
        )
        parser.add_argument(
            "--noisy-dir",
            "-i",
            type=str,
            default=None,
            help="Input directory containing noisy audio files. Use instead of `noisy_audio_files`.",
        )
        parser.add_argument(
            "--no-suffix",
            action="store_false",
            dest="suffix",
            help="Don't add the model suffix to the enhanced audio files",
        )
        parser.add_argument("--no-df-stage", action="store_true")
        args = parser.parse_args()
        main_dfnet(args)

    ########################################################################
    
    def pub_loop(self):
        # time.sleep(5)   
        # self.input_pub.publish('Program started')
        while True:
            time.sleep(1)
            if self.user_input != '':
                self.input_pub.publish(self.user_input)
                # print(f'{self.user_input}')
                self.user_input = ''

    def handle_user_input(self, input_data):
        self.user_input = input_data

    def handle_audio_input(self, input_data):
        self.audio_input = input_data

    def get_msg_class(self, msg_type):
        """
        Given a ROS message type specified as a string, e.g.
            "std_msgs/Int32"
        or
            "std_msgs/msg/Int32"
        it imports the message class into Python and returns the class, i.e. the actual std_msgs.msg.Int32
        
        Returns none if the type is invalid (e.g. if user hasn't bash-sourced the message package).
        """
        try:
            msg_module, dummy, msg_class_name = msg_type.replace("/", ".").rpartition(".")
        except ValueError:
            rospy.logerr("invalid type %s" % msg_type)
            return None

        try:
            if not msg_module.endswith(".msg"):
                msg_module = msg_module + ".msg"
            return getattr(importlib.import_module(msg_module), msg_class_name)
        except Exception as e:
            rospy.logerr(str(e))
            return None

    def pingpong_loop(self):
        """
        Loop to send pings to all active sockets every 5 seconds.
        """
        while True:
            time.sleep(5)

            if self.event_loop is None:
                continue
            try:
                self.event_loop.add_callback(ROSBoardSocketHandler.send_pings)
            except Exception as e:
                rospy.logwarn(str(e))
                traceback.print_exc()

    def sync_subs_loop(self):
        """
        Periodically calls self.sync_subs(). Intended to be run in a thread.
        """
        while True:
            time.sleep(1)
            self.sync_subs()

    def sync_subs(self):
        """
        Looks at self.remote_subs and makes sure local subscribers exist to match them.
        Also cleans up unused local subscribers for which there are no remote subs interested in them.
        """

        # Acquire lock since either sync_subs_loop or websocket may call this function (from different threads)
        self.lock.acquire()

        try:
            # all topics and their types as strings e.g. {"/foo": "std_msgs/String", "/bar": "std_msgs/Int32"}
            self.all_topics = {}

            for topic_tuple in rospy.get_published_topics():
                topic_name = topic_tuple[0]
                topic_type = topic_tuple[1]
                if type(topic_type) is list:
                    topic_type = topic_type[0] # ROS2
                self.all_topics[topic_name] = topic_type

            self.event_loop.add_callback(
                ROSBoardSocketHandler.broadcast,
                [ROSBoardSocketHandler.MSG_TOPICS, self.all_topics ]
            )

            for topic_name in self.remote_subs:
                if len(self.remote_subs[topic_name]) == 0:
                    continue

                # remote sub special (non-ros) topic: _dmesg
                # handle it separately here
                if topic_name == "_dmesg":
                    if topic_name not in self.local_subs:
                        rospy.loginfo("Subscribing to dmesg [non-ros]")
                        self.local_subs[topic_name] = DMesgSubscriber(self.on_dmesg)
                    continue

                if topic_name == "_system_stats":
                    if topic_name not in self.local_subs:
                        rospy.loginfo("Subscribing to _system_stats [non-ros]")
                        self.local_subs[topic_name] = SystemStatsSubscriber(self.on_system_stats)
                    continue

                if topic_name == "_top":
                    if topic_name not in self.local_subs:
                        rospy.loginfo("Subscribing to _top [non-ros]")
                        self.local_subs[topic_name] = ProcessesSubscriber(self.on_top)
                    continue

                # check if remote sub request is not actually a ROS topic before proceeding
                if topic_name not in self.all_topics:
                    rospy.logwarn("warning: topic %s not found" % topic_name)
                    continue

                # if the local subscriber doesn't exist for the remote sub, create it
                if topic_name not in self.local_subs:
                    topic_type = self.all_topics[topic_name]
                    msg_class = self.get_msg_class(topic_type)

                    if msg_class is None:
                        # invalid message type or custom message package not source-bashed
                        # put a dummy subscriber in to avoid returning to this again.
                        # user needs to re-run rosboard with the custom message files sourced.
                        self.local_subs[topic_name] = DummySubscriber()
                        self.event_loop.add_callback(
                            ROSBoardSocketHandler.broadcast,
                            [
                                ROSBoardSocketHandler.MSG_MSG,
                                {
                                    "_topic_name": topic_name, # special non-ros topics start with _
                                    "_topic_type": topic_type,
                                    "_error": "Could not load message type '%s'. Are the .msg files for it source-bashed?" % topic_type,
                                },
                            ]
                        )
                        continue

                    self.last_data_times_by_topic[topic_name] = 0.0

                    rospy.loginfo("Subscribing to %s" % topic_name)

                    self.local_subs[topic_name] = rospy.Subscriber(
                        topic_name,
                        self.get_msg_class(topic_type),
                        self.on_ros_msg,
                        callback_args = (topic_name, topic_type),
                    )

            # clean up local subscribers for which remote clients have lost interest
            for topic_name in list(self.local_subs.keys()):
                if topic_name not in self.remote_subs or \
                    len(self.remote_subs[topic_name]) == 0:
                        rospy.loginfo("Unsubscribing from %s" % topic_name)
                        self.local_subs[topic_name].unregister()
                        del(self.local_subs[topic_name])

        except Exception as e:
            rospy.logwarn(str(e))
            traceback.print_exc()
        
        self.lock.release()

    def on_system_stats(self, system_stats):
        """
        system stats received. send it off to the client as a "fake" ROS message (which could at some point be a real ROS message)
        """
        if self.event_loop is None:
            return

        msg_dict = {
            "_topic_name": "_system_stats", # special non-ros topics start with _
            "_topic_type": "rosboard_msgs/msg/SystemStats",
        }

        for key, value in system_stats.items():
            msg_dict[key] = value

        self.event_loop.add_callback(
            ROSBoardSocketHandler.broadcast,
            [
                ROSBoardSocketHandler.MSG_MSG,
                msg_dict
            ]
        )

    def on_top(self, processes):
        """
        processes list received. send it off to the client as a "fake" ROS message (which could at some point be a real ROS message)
        """
        if self.event_loop is None:
            return

        self.event_loop.add_callback(
            ROSBoardSocketHandler.broadcast,
            [
                ROSBoardSocketHandler.MSG_MSG,
                {
                    "_topic_name": "_top", # special non-ros topics start with _
                    "_topic_type": "rosboard_msgs/msg/ProcessList",
                    "processes": processes,
                },
            ]
        )

    def on_dmesg(self, text):
        """
        dmesg log received. make it look like a rcl_interfaces/msg/Log and send it off
        """
        if self.event_loop is None:
            return

        self.event_loop.add_callback(
            ROSBoardSocketHandler.broadcast,
            [
                ROSBoardSocketHandler.MSG_MSG,
                {
                    "_topic_name": "_dmesg", # special non-ros topics start with _
                    "_topic_type": "rcl_interfaces/msg/Log",
                    "msg": text,
                },
            ]
        )

    def on_ros_msg(self, msg, topic_info):
        """
        ROS messaged received (any topic or type).
        """
        topic_name, topic_type = topic_info
        t = time.time()
        if t - self.last_data_times_by_topic.get(topic_name, 0) < self.update_intervals_by_topic[topic_name] - 1e-4:
            return

        if self.event_loop is None:
            return

        # convert ROS message into a dict and get it ready for serialization
        ros_msg_dict = ros2dict(msg)

        # add metadata
        ros_msg_dict["_topic_name"] = topic_name
        ros_msg_dict["_topic_type"] = topic_type
        ros_msg_dict["_time"] = time.time() * 1000

        # log last time we received data on this topic
        self.last_data_times_by_topic[topic_name] = t

        # broadcast it to the listeners that care    
        self.event_loop.add_callback(
            ROSBoardSocketHandler.broadcast,
            [ROSBoardSocketHandler.MSG_MSG, ros_msg_dict]
        )

def main(args=None):
    ROSBoardNode().start()

if __name__ == '__main__':
    main()


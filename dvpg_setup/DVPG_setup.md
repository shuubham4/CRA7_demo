# Accessing specific web pages in CARDS data server through SSH
The following tutorial can be followed irrespective of operating systems (Windows Subsystem for Linux (WSL) may need to be installed in older Windows versions).
- Download Firefox and install (if not installed already)
- Open command prompt/terminal and type the following:

ssh -N -D <port number> <username>@130.85.151.34 -p 2002

It will ask for user password and upon providing that it will remain in standby until you press ctrl-C.

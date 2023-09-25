import os
import sys
import socket
from socket import socket,AF_INET,SOCK_STREAM
platformSock = socket(AF_INET,SOCK_STREAM)
platformSock.connect(('localhost',2322))
#Power on the tpm
platformSock.send('\0\0\0\1')
tpmSock = socket(AF_INET,SOCK_STREAM)
tpmSock.connect(('localhost',2321))
#Send tpm_Send_command
tpmSock.send('\x00\x00\x00\x08')
#Send locality
tpmSock.send('\x03')
#Send # of bytes
tpmSock.send('\x00\x00\x00\x0c')
#Send tag
tpmSock.send('\x80\x01')
#Send command size
tpmSock.send('\x00\x00\x00\x0c')
#Send command code TPM_Startup
tpmSock.send('\x00\x00\x01\x44')
#Send TPM SU
tpmSock.send('\x00\x00')
#receive the size of response,the response, and 4 bytes of 0's
reply=tpmSock.recv(18)
for c in reply:
    print ("%#x " % ord(c))
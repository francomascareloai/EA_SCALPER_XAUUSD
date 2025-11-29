---
title: "Network functions"
url: "https://www.mql5.com/en/docs/network"
hierarchy: []
scraped_at: "2025-11-28 09:31:00"
---

# Network functions

[MQL5 Reference](/en/docs "MQL5 Reference")Network Functions

* [SocketCreate](/en/docs/network/socketcreate "SocketCreate")
* [SocketClose](/en/docs/network/socketclose "SocketClose")
* [SocketConnect](/en/docs/network/socketconnect "SocketConnect")
* [SocketIsConnected](/en/docs/network/socketisconnected "SocketIsConnected")
* [SocketIsReadable](/en/docs/network/socketisreadable "SocketIsReadable")
* [SocketIsWritable](/en/docs/network/socketiswritable "SocketIsWritable")
* [SocketTimeouts](/en/docs/network/sockettimeouts "SocketTimeouts")
* [SocketRead](/en/docs/network/socketread "SocketRead")
* [SocketSend](/en/docs/network/socketsend "SocketSend")
* [SocketTlsHandshake](/en/docs/network/sockettlshandshake "SocketTlsHandshake")
* [SocketTlsCertificate](/en/docs/network/sockettlscertificate "SocketTlsCertificate")
* [SocketTlsRead](/en/docs/network/sockettlsread "SocketTlsRead")
* [SocketTlsReadAvailable](/en/docs/network/sockettlsreadavailable "SocketTlsReadAvailable")
* [SocketTlsSend](/en/docs/network/sockettlssend "SocketTlsSend")
* [WebRequest](/en/docs/network/webrequest "WebRequest")
* [SendFTP](/en/docs/network/sendftp "SendFTP")
* [SendMail](/en/docs/network/sendmail "SendMail")
* [SendNotification](/en/docs/network/sendnotification "SendNotification")

# Network functions

MQL5 programs can exchange data with remote servers, as well as send push notifications, emails and data via FTP.

* The [Socket\*](/en/docs/network/socketcreate) group of functions allows establishing a TCP connection (including a secure TLS) with a remote host via system sockets. The operation principle is simple: [create a socket](/en/docs/network/socketcreate), [connect to the server](/en/docs/network/socketconnect) and start [reading](/en/docs/network/socketread) and [writing](/en/docs/network/socketsend) data.
* The [WebRequest](/en/docs/network/webrequest) function is designed to work with web resources and allows sending HTTP requests (including GET and POST) easily.
* [SendFTP](/en/docs/network/sendftp), [SendMail](/en/docs/network/sendmail) and [SendNotification](/en/docs/network/sendnotification) are more simple functions for sending files, emails and mobile notifications.

For end-user security, the list of allowed IP addresses is implemented on the client terminal side. The list contains IP addresses the MQL5 program is allowed to connect to via the Socket\* and WebRequest functions. For example, if the program needs to connect to https://www.someserver.com, this address should be explicitly indicated by a terminal user in the list. An address cannot be added programmatically.

![Adding the address to the list](/en/docs/img/allow_net_request.png "Adding the address to the list")

Add an explicit message to the MQL5 program to notify a user of the need for additional configuration. You can do that via [#property description](/en/docs/basis/preprosessor/compilation), [Alert](/en/docs/common/alert) or [Print](/en/docs/common/print).

| Function | Action |
| --- | --- |
| [SocketCreate](/en/docs/network/socketcreate) | Create a socket with specified flags and return its handle |
| [SocketClose](/en/docs/network/socketclose) | Close a socket |
| [SocketConnect](/en/docs/network/socketconnect) | Connect to the server with timeout control |
| [SocketIsConnected](/en/docs/network/socketisconnected) | Checks if the socket is currently connected |
| [SocketIsReadable](/en/docs/network/socketisreadable) | Get a number of bytes that can be read from a socket |
| [SocketIsWritable](/en/docs/network/socketiswritable) | Check whether data can be written to a socket at the current time |
| [SocketTimeouts](/en/docs/network/sockettimeouts) | Set timeouts for receiving and sending data for a socket system object |
| [SocketRead](/en/docs/network/socketread) | Read data from a socket |
| [SocketSend](/en/docs/network/socketsend) | Write data to a socket |
| [SocketTlsHandshake](/en/docs/network/sockettlshandshake) | Initiate secure TLS (SSL) connection to a specified host via TLS Handshake protocol |
| [SocketTlsCertificate](/en/docs/network/sockettlscertificate) | Get data on the certificate used to secure network connection |
| [SocketTlsRead](/en/docs/network/sockettlsread) | Read data from secure TLS connection |
| [SocketTlsReadAvailable](/en/docs/network/sockettlsreadavailable) | Read all available data from secure TLS connection |
| [SocketTlsSend](/en/docs/network/sockettlssend) | Send data via secure TLS connection |
| [WebRequest](/en/docs/network/webrequest) | Send an HTTP request to a specified server |
| [SendFTP](/en/docs/network/sendftp) | Send a file to an address specified on the FTP tab |
| [SendMail](/en/docs/network/sendmail) | Send an email to an address specified in the Email tab of the options window |
| [SendNotification](/en/docs/network/sendnotification) | Send push notifications to mobile terminals whose MetaQuotes IDs are specified in the Notifications tab |

[SignalUnSubscribe](/en/docs/signals/signalunsubscribe "SignalUnSubscribe")

[SocketCreate](/en/docs/network/socketcreate "SocketCreate")
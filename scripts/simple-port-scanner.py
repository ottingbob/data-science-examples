import ftplib
import socket

# Scripts for gathering info on devices.
# Useful for getting scans on a local setup etc.


def conn_scan(host: str, port: int):
    conn_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        conn_sock.connect((host, port))
        print(f"[+] {port}/tcp open")
        conn_sock.close()
    except TimeoutError:
        print(f"[-] {port}/tcp closed")


def port_scan(host: str, ports: int):
    ip = socket.gethostbyname(host)
    name = socket.gethostbyaddr(ip)
    print(f"Scan result of {name[0]}")
    socket.setdefaulttimeout(1)
    for port in ports:
        print("Scanning port:", port)
        conn_scan(host, port)


# Perform an FTP scan
def ftp_anon_login(host: str):
    try:
        ip = socket.gethostbyname(host)
        ftp = ftplib.FTP(ip)
        ftp.login("anonymous")
        print(f"\n[+] {host} FTP Anonymous Login Succeeded")
    except Exception:
        print(f"\n[-] {host} FTP Anonymous Login Failed")


port_scan("google.com", [80, 22])
ftp_anon_login("speedtest.tele2.net")

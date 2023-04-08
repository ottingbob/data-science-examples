import os
from pathlib import Path

import scapy.all
import scapy.layers.inet
import scapy.utils
from geoip2.database import Reader as gipReader
from scapy.plist import PacketList

ROOT_DIR = str(Path(__file__).parent.parent)

# Get the GeoLiteCity data file from:
# http://geolite.maxmind.com/download/geoip/database/GeoLiteCity.dat.gz
data_file = f"GeoLite2-City_20230407{os.sep}GeoLite2-City.mmdb"
GEOIP_CITY_FILE = Path(ROOT_DIR + os.sep + data_file)


# Attach a geo location to our IPs
def ret_kml(dst_ip, src_ip, geo_reader):
    # Random datacenter in Chicago
    SRC_IP = "38.145.192.6"
    try:
        dst = geo_reader.city(dst_ip).location
        src = geo_reader.city(SRC_IP).location

        dst_longitude = dst.longitude
        dst_latitude = dst.latitude
        src_longitude = src.longitude
        src_latitude = src.latitude

        if dst_latitude is None or dst_longitude is None:
            return ""
        if src_longitude is None or src_latitude is None:
            return ""

        kml = (
            "<Placemark>\n"
            f"<name>{dst_ip}</name>\n"
            "<extrude>1</extrude>\n"
            "<tessellate>1</tessellate>\n"
            "<styleUrl>#transBluePoly</styleUrl>\n"
            "<LineString>\n"
            "<coordinates>\n"
            f"{dst_longitude:6f},{dst_latitude:6f}\n"
            f"{src_longitude:6f},{src_latitude:6f}\n"
            "</coordinates>\n"
            "</LineString>\n"
            "</Placemark>\n"
        )
        return kml
    except Exception as err:
        print(f"Encountered error during ip parsing: {err}")
        return ""


# Loop over packet capture and extract the IPs
def plot_ips(pcap: PacketList):
    kml_pts = ""
    # Prints total number of TCP / UDP / ICMP packets in list
    print(pcap)
    with gipReader(GEOIP_CITY_FILE) as reader:
        for packet in pcap:
            try:
                ip = packet.payload
                # The geoIP reader won't be able to find local IP addresses...
                if str(ip.dst).startswith("192.168"):
                    continue

                kml = ret_kml(ip.dst, ip.src, reader)
                kml_pts += kml

                # Unsure how to parse the IP parts out
                # print(packet.payload)
                # print(scapy.layers.inet.Ether(packet))
                # print(packet.sprintf("%IP.src%"), packet.sprintf("%IP.dst%"))
            except Exception as err:
                print(f"Error reading packet payload: {err}")
    return kml_pts


# Read in the wire pcap file
pcap_file = Path(f"{ROOT_DIR}{os.sep}wire.pcap")
kml_output_file = Path(f"{ROOT_DIR}{os.sep}network_capture_output.kml")
with open(pcap_file, "rb") as pcap_reader:
    pcap = scapy.utils.rdpcap(pcap_reader)
    kml_header = (
        '<?xml version="1.0" encoding="UTF-8"?> \n<kml xmlns="http://www.opengis.net/kml/2.2">\n<Document>\n'
        '<Style id="transBluePoly">'
        "<LineStyle>"
        "<width>1.5</width>"
        "<color>501400E6</color>"
        "</LineStyle>"
        "</Style>"
    )
    kml_footer = "</Document>\n</kml>\n"
    kml_doc = kml_header + plot_ips(pcap) + kml_footer

    # Write the network capture keyhole file
    # Take this file to google earth and visualize the connections!
    # https://earth.google.com/web/
    # 1) Projects
    # 2) Open > Import KML file from computer
    # 3) $$$
    f = open(kml_output_file, "w")
    f.write(kml_doc)
    f.close()

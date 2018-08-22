import threading

from firebase import get_firebase_ref

from ObjectDetection import ObjectDetection
from BaseStation import BaseStation
from WebServer import WebServer

(USER, FIREBASE_REF) = get_firebase_ref()


def main():
    object_detection = ObjectDetection(camera=0, model_name='ssd_mobilenet_v2_coco_2018_03_29',
                                       label_path="./label_maps/mscoco_label_map.pbtxt", detect_class=90)
    object_detection.begin_detection()
    """ base_station = BaseStation(USER, FIREBASE_REF, object_detection)
    web_server = WebServer(base_station)

    web_server.start() """


if __name__ == '__main__':
    main()

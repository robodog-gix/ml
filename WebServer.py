import threading

from flask import Flask, render_template, Response


class WebServer:
    def __init__(self, base_station):
        self.bs = base_station  # generator function that yields image

        self.app = Flask(__name__)
        self.app.add_url_rule("/", "index", self.index)
        self.app.add_url_rule("/video_feed", "video_feed", self.video_feed)
        self.app.add_url_rule("/launch", "launch", self.launch)

    def index(self):
        return render_template('index.html')

    def video_feed(self):
        return Response(self.bs.od.begin_detection(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    def launch(self):
        launch_thread = threading.Thread()
        launch_thread.start()
        return Response('Launch!', 200)

    def start(self):
        self.app.run(host='0.0.0.0', debug=False)

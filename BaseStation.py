import logging
import time

# Only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)


class BaseStation():
    def __init__(self, user, db, od):
        self.user = user  # credentials for writing to firebase
        self.db = db  # firebase reference
        self.od = od  # object detection reference

        # initialize state in firebase
        """ self.db.update({
            "object": "NONE"
        }, self.user['idToken'])   """
""" 
    def mock_state_machine(self):
        self.db.update({"state": "LAUNCH"}, self.user['idToken'])
        self.sleep_2()
        self.db.update({"state": "SEARCH"}, self.user['idToken'])
        self.sleep_2()
        self.db.update({"state": "APPROACH"}, self.user['idToken'])
        self.sleep_2()
        self.db.update({"state": "DETER"}, self.user['idToken']) """

    

def main():
    base_station = BaseStation()
    user_interface = UserInterface(base_station)
    user_interface.start()


if __name__ == '__main__':
    main()

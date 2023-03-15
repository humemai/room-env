import unittest

from room_env.des import RoomDes


class DesTest(unittest.TestCase):
    def test_all(self) -> None:
        for des_size in ["xxs", "xs", "s", "m", "l"]:
            des = RoomDes(des_size=des_size)
            des.run()

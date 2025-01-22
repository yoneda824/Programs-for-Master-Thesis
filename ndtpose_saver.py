# coding:utf-8

import csv
import os
import numpy as np

import rospy
from geometry_msgs.msg import PoseStamped

# frequency [Hz]
FREQUENCY = 10  # データ記録の更新頻度 [Hz]
SPACING = 0  # ウェイポイント間の最小距離
HOME_DIR = os.path.expanduser("~")  # ホームディレクトリのパス
FILE_PATH = os.path.join(HOME_DIR, "catkin_crawler/autodrive/ndt_pose.csv")  # ウェイポイント保存先のファイルパス

class waypoint2csv():
    def __init__(self):
        # ROSノードを初期化
        rospy.init_node('ndtpose_saver')
        # '/ndt_pose'トピックからPoseStampedメッセージを購読
        rospy.Subscriber('/ndt_pose', PoseStamped, self.odomCallback)

        # シャットダウン時の処理を登録
        rospy.on_shutdown(self.shutdown)

    def shutdown(self):
        # ファイルを閉じる
        self.f.close()
        rospy.loginfo("shutdown")

    def odomCallback(self, msg):
        # 現在の位置情報を更新 (X座標, Y座標)
        self.x = msg.pose.position.x
        self.y = msg.pose.position.y
        #print(self.x, self.y)

    def loop(self):
        rate = rospy.Rate(FREQUENCY)  # ループの更新頻度を設定
        last_x = None  # 前回記録したX座標
        last_y = None  # 前回記録したY座標

        # ファイルが存在しない場合、新規作成してヘッダーを追加
        file_exists = os.path.isfile(FILE_PATH)
        if not file_exists:
            self.f = open(FILE_PATH, 'a')
            csvWriter = csv.writer(self.f)
            csvWriter.writerow(['field.pose.position.x', 'field.pose.position.y'])  # ヘッダー行を追加
            self.f.close()

        while not rospy.is_shutdown():
            # 初回実行時、前回座標を初期化
            if last_x is None or last_y is None:
                try:
                    last_x = self.x
                    last_y = self.y
                    rospy.loginfo("last_x and last_y initialized")
                except:
                    continue

            # 現在の座標と前回の座標の間の距離を計算
            a = np.array([last_x, last_y])
            b = np.array([self.x, self.y])
            d = np.linalg.norm(a - b)  # 距離を計算

            # ウェイポイント間の距離が設定値以上の場合に記録
            if abs(d) > SPACING:
                self.f = open(FILE_PATH, 'a')
                csvWriter = csv.writer(self.f)

                # 現在の座標をCSVに記録
                csvWriter.writerow([self.x, self.y])
                print("x=", self.x, "y=", self.y, "d=", d)

                # 前回の座標を更新
                last_x = self.x
                last_y = self.y
            else:
                # 距離が閾値未満の場合、記録をスキップ
                print("Waypoint save skipped", "d:", d)

            rate.sleep()  # 指定した更新頻度で待機
    
if __name__ == '__main__':
    w = waypoint2csv()  # waypoint2csvクラスのインスタンスを作成
    w.loop()  # データ記録ループを開始

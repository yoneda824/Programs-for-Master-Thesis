#! /usr/bin/python
# coding:utf-8

import os
from datetime import datetime
import rospy
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import binary_dilation
import heapq
import csv
import time
import load_waypoint
import tf2_ros
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from tf.transformations import quaternion_from_euler
from tf.transformations import euler_from_quaternion
from tf.transformations import quaternion_multiply

import rosnode
import subprocess
# ros custom message
from crawler_look_ahead.msg import Lookahead_Log

HOME_DIR = os.path.expanduser("~")
DATE = datetime.now()
DATE_LABEL = DATE.strftime("%Y%m%d_%H%M%S_")

rospy.init_node('autodrive_following')

FILE_PATH = os.path.join(HOME_DIR, "catkin_crawler/autodrive/ndt_pose.csv")

ROUTE_NAME = DATE_LABEL + rospy.get_param('~logname_param') + "_route.csv"
ROUTE_FILE_PATH = os.path.join(HOME_DIR, "catkin_crawler/autodrive/", ROUTE_NAME)

LOG_NAME = DATE_LABEL + rospy.get_param('~logname_param')
LOG_FILE_PATH = os.path.join(HOME_DIR, "catkin_crawler/autodrive/log", LOG_NAME)

# CSVファイルの読み込み（ヘッダー付き）
data = pd.read_csv(FILE_PATH)
points = data[['field.pose.position.x', 'field.pose.position.y']].values

# 座標の補正（負の値を補正）
x_min = np.min(points[:, 0])
y_min = np.min(points[:, 1])
points[:, 0] = points[:, 0] - x_min if x_min < 0 else points[:, 0]
points[:, 1] = points[:, 1] - y_min if y_min < 0 else points[:, 1]

# 補正量を表示
print("x方向の補正量: {:.2f}".format(-x_min if x_min < 0 else 0))
print("y方向の補正量: {:.2f}".format(-y_min if y_min < 0 else 0))

# 点群データの平滑化
points[:, 0] = gaussian_filter1d(points[:, 0], sigma=2)
points[:, 1] = gaussian_filter1d(points[:, 1], sigma=2)

# スケーリングファクター
scale_factor = 20

# 座標をスケーリング
scaled_points = points * scale_factor

# マップの設定（解像度を1000倍に）
width = int(np.max(scaled_points[:, 0])) + 1
height = int(np.max(scaled_points[:, 1])) + 1
radius = 0  # コストを0に設定する範囲の半径（ピクセル）

# コストマップの初期化（すべてのセルを最高コストで初期化）
cost_map = np.full((height, width), 255, dtype=np.uint8)

# 線形補間で点と点の間を埋める
def interpolate_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    distance = max(abs(x2 - x1), abs(y2 - y1))
    points = []
    for i in range(int(distance) + 1):  # '+1' to ensure inclusive of end point
        t = i / float(distance)
        x = (1 - t) * x1 + t * x2
        y = (1 - t) * y1 + t * y2
        points.append((int(round(x)), int(round(y))))  # Use round to ensure integer values
    return points

# 近い点のペアを見つける関数
def find_close_points(points, distance_threshold):
    close_pairs = []
    num_points = len(points)
    for i in range(num_points):
        for j in range(i + 1, num_points):
            distance = np.linalg.norm(points[i] - points[j])
            if distance < distance_threshold:
                close_pairs.append((i, j))
    return close_pairs

# スケーリングされた点群データに基づいて周囲のコストを低く設定
for i in range(len(scaled_points) - 1):
    point1 = scaled_points[i]
    point2 = scaled_points[i + 1]
    interpolated_points = interpolate_points(point1, point2)
    
    # ピクセル間をつなぐ処理
    for x, y in interpolated_points:
        if 0 <= x < width and 0 <= y < height:
            # 1ピクセル幅で線をつなぐ
            cost_map[y, x] = 0

# 膨張処理を追加して道幅を広げる
dilation_iterations = 1  # 道幅を広げる程度（調整可能）
expanded_map = binary_dilation(cost_map == 0, iterations=dilation_iterations)

# コストマップを更新（0: 通行可能、255: 通行不可）
cost_map = np.where(expanded_map, 0, 255).astype(np.uint8)


# PGMファイルとして保存（デバッグ用）
image = Image.fromarray(cost_map)
PGM_NAME = DATE_LABEL + rospy.get_param('~logname_param') + "_costmap.pgm"
pgm_path = os.path.join(HOME_DIR, "catkin_crawler/autodrive/calc/", PGM_NAME)
image.save(pgm_path)

# コストマップを表示してクリックした2点を取得する
# コストマップを表示してクリックしたゴール地点を取得する
def get_start_and_goal(cost_map, start_point):
    """
    コストマップを表示してゴール地点をクリックで取得する。
    スタート地点は自動設定。
    """
    plt.imshow(cost_map, cmap='gray')  # コストマップを表示
    plt.gca().invert_yaxis()  # 0点を左下に設定
    plt.title('Click goal point (Start is already set)')

    # スタート地点を青い丸で表示
    plt.scatter(start_point[0], start_point[1], color='blue', s=100, label='Start Point', edgecolor='black')
    
    # ゴール地点をクリックして指定（無制限に待機）
    points = plt.ginput(n=1, timeout=0)  # 1点クリック、無制限に待機
    
    # ゴール地点を赤い丸で表示
    if points:  # ユーザーがクリックした場合のみ処理
        goal_point = (int(points[0][0]), int(points[0][1]))
        plt.scatter(goal_point[0], goal_point[1], color='red', s=100, label='Goal Point', edgecolor='black')
        print("Selected Goal Point: ({:.2f}, {:.2f})".format(goal_point[0], goal_point[1]))
    else:
        print("No goal point was selected.")
        plt.close()
        return None, None  # クリックが行われなかった場合はNoneを返す
    
    plt.legend()  # 凡例を追加
    plt.show()  # クリック後のマップを表示

    return start_point, goal_point


# コスト0の点を全てリスト化
def get_cost_zero_points(cost_map, csv_file_path):
    zero_points = np.argwhere(cost_map == 0)  # コスト0の全ての点の座標を取得 (y, x)
    # リスト内の順番を (x, y) に変更
    zero_points_xy = [(point[1], point[0]) for point in zero_points]  # (x, y)形式に変換
    # コスト0の点をCSVとして保存
    with open(csv_file_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['x','y'])  # ヘッダー（y, x 順）
        for point in zero_points_xy:
            writer.writerow([point[0], point[1]])  # y, x 形式で保存
    
    print("Cost 0 points saved to", csv_file_path)
    
    return zero_points_xy

# コスト0の点を取得し、CSVに保存
#csv_file_path = 'cost_zero_points.csv' 
COST_CSV_NAME = DATE_LABEL + rospy.get_param('~logname_param') + "_cost_zero_points.csv"
csv_file_path = os.path.join(HOME_DIR, "catkin_crawler/autodrive/calc/", COST_CSV_NAME)
cost_zero_points = get_cost_zero_points(cost_map, csv_file_path)

# クリックした点に最も近いコスト0の点を探す
def find_nearest_cost_zero_point(clicked_point, zero_points):
    tree = KDTree(zero_points)
    distance, index = tree.query(clicked_point)
    nearest_point = zero_points[index]
    return nearest_point  # (x, y) の形式で返す

# ndt_pose.csvの最後の行からスタート地点を取得
start_point = (int(scaled_points[-1][0]), int(scaled_points[-1][1]))  # スケール済みの最後の座標を取得

# ゴール地点の指定（スタート地点は自動取得）
start, goal = get_start_and_goal(cost_map, start_point)

# ゴールが選択されなかった場合の処理
if start is None or goal is None:
    print("Start or Goal point not properly selected. Exiting...")
    exit(0)

# スタート地点とゴール地点に最も近いコスト0の点を探して設定
start = find_nearest_cost_zero_point(start, cost_zero_points)
goal = find_nearest_cost_zero_point(goal, cost_zero_points)

print("Nearest start:", start)
print("Nearest goal:", goal)


# ベクトル間の角度を計算する関数
def angle_between_vectors(v1, v2):
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # cosθをクリップして安全な範囲に
    return math.degrees(angle)  # ラジアンを度に変換


# A*アルゴリズムの実装
grid = np.array(cost_map)

class Node:
    def __init__(self, x, y, cost=0.0, parent=None):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent = parent
        self.heuristic = 0.0
        self.total_cost = 0.0

    def __lt__(self, other):
        return self.total_cost < other.total_cost

def is_navigable(x, y):
    return grid[int(y)][int(x)] == 0  # y, x の順番でアクセスする

def get_neighbors(node):
    neighbors = []
    step_size = 1  # ステップサイズを1に設定
    directions = [(step_size, 0), (-step_size, 0), (0, step_size), (0, -step_size), 
                  (step_size, step_size), (-step_size, step_size), (step_size, -step_size), (-step_size, -step_size)]
    for dx, dy in directions:
        nx, ny = node.x + dx, node.y + dy
        if 0 <= int(nx) < grid.shape[1] and 0 <= int(ny) < grid.shape[0] and is_navigable(nx, ny):
            new_cost = node.cost + ((dx ** 2 + dy ** 2) ** 0.5)  # ユークリッド距離をコストとして使用
            neighbors.append(Node(nx, ny, new_cost, node))
    return neighbors

# 各列の中央を計算
def calculate_column_centers(grid):
    centers = {}
    for x in range(grid.shape[1]):
        navigable_points = np.where(grid[:, x] == 0)[0]
        if len(navigable_points) > 0:
            centers[x] = np.mean(navigable_points)
    return centers

column_centers = calculate_column_centers(grid)

# ヒューリスティック関数
def heuristic(x1, y1, x2, y2, prev_node=None):
    goal_distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5  # ユークリッド距離
    if prev_node is not None:
        prev_vector = (x1 - prev_node.x, y1 - prev_node.y)
        next_vector = (x2 - x1, y2 - y1)
        angle_change = angle_between_vectors(prev_vector, next_vector)
    else:
        angle_change = 0  # 最初のノードは角度の変化を考慮しない
    return goal_distance +100.0 * angle_change  # 角度の変化を加味したヒューリスティック

def save_route(route, file_path):
    try:
        with open(file_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y'])  # Header
            for x, y in route:
                x_scaled = (x / scale_factor) + x_min if x_min < 0 else (x / scale_factor)
                y_scaled = (y / scale_factor) + y_min if y_min < 0 else (y / scale_factor)
                writer.writerow(["{:.2f}".format(x_scaled), "{:.2f}".format(y_scaled)])  # 小数点以下2桁で出力
        print("File successfully saved to", file_path)
    except Exception as e:
        print("Failed to save file:", str(e))

def linear_interpolate(points):
    """
    経路の線形補間
    """
    interpolated_path = []
    for i in range(len(points) - 1):
        start, end = points[i], points[i + 1]
        steps = int(max(abs(end[0] - start[0]), abs(end[1] - start[1])))
        for step in range(steps):
            t = step / float(steps)
            x = (1 - t) * start[0] + t * end[0]
            y = (1 - t) * start[1] + t * end[1]
            interpolated_path.append((x, y))
    interpolated_path.append(points[-1])
    return interpolated_path

def smooth_path(points, weight_data=0.5, weight_smooth=0.1, tolerance=0.00001): 
    """
    経路を滑らかにするためのスムージング関数
    """
    new_path = np.array(points)
    change = tolerance
    while change >= tolerance:
        change = 0.0
        for i in range(1, len(points) - 1):
            for j in range(2):
                aux = new_path[i][j]
                new_path[i][j] += weight_data * (points[i][j] - new_path[i][j])
                new_path[i][j] += weight_smooth * (new_path[i - 1][j] + new_path[i + 1][j] - 2.0 * new_path[i][j])
                change += abs(aux - new_path[i][j])
    return new_path.tolist()

def a_star(start, goal, save_path):
    open_set = []
    closed_set = set()
    start_node = Node(start[0], start[1])
    goal_node = Node(goal[0], goal[1])
    heapq.heappush(open_set, (0, start_node))

    while open_set:
        current_cost, current_node = heapq.heappop(open_set)

        if abs(current_node.x - goal_node.x) <= 1 and abs(current_node.y - goal_node.y) <= 1:
            path = []
            while current_node:
                path.append((current_node.x, current_node.y))
                current_node = current_node.parent
            interpolated_path = linear_interpolate(path[::-1])  # 経路を逆転させて線形補間
            smoothed_path = smooth_path(interpolated_path)  # 経路をスムージング
            save_route(smoothed_path, save_path)
            print("Route found and saved.")
            return smoothed_path

        closed_set.add((current_node.x, current_node.y))

        neighbors = get_neighbors(current_node)
        for neighbor in neighbors:
            if (neighbor.x, neighbor.y) in closed_set:
                continue
            if (neighbor.x, neighbor.y) not in [(n.x, n.y) for _, n in open_set]:  # 既存のオープンセットに存在しない場合のみ追加
                neighbor.heuristic = heuristic(neighbor.x, neighbor.y, goal_node.x, goal_node.y)
                neighbor.total_cost = neighbor.cost + neighbor.heuristic
                heapq.heappush(open_set, (neighbor.total_cost, neighbor))
                print("Exploring node at ({:.2f}, {:.2f}) with cost {:.2f}, heuristic {:.2f}".format(current_node.x, current_node.y, current_cost, neighbor.heuristic))

    print("No path found.")
    return None

# 新しいCSVファイル名を生成し、保存先のパスを指定する
NEW_CSV_NAME = DATE_LABEL + rospy.get_param('~logname_param') + "_saved_route_new.csv"
save_path = os.path.join(HOME_DIR, "catkin_crawler/autodrive/calc/", NEW_CSV_NAME)

# 経路を反転して反復経路を作成する関数
def create_round_trip_path(path):
    # 反転して往復経路を作成する
    return_path = path[::-1]  # 経路を反転
    round_trip_path = path + return_path[1:-1]  # 最初の点の重複を避ける
    return round_trip_path

# 経路ファイルを読み込み
def load_route(file_path):
    route = []
    try:
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # ヘッダーをスキップ
            for row in reader:
                x, y = float(row[0]), float(row[1])
                route.append((x, y))
        print("Route successfully loaded from", file_path)
    except Exception as e:
        print("Failed to load route:", str(e))
    return route

# 反復経路を作成してCSVファイルを更新
def save_round_trip_route(route, file_path):
    try:
        with open(file_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y'])  # ヘッダー
            for x, y in route:
                writer.writerow(["{:.2f}".format(x), "{:.2f}".format(y)])  # 小数点以下2桁で保存
        print("Route successfully updated with round trip.")
    except Exception as e:
        print("Failed to save round trip route:", str(e))

# 経路ファイルを出力して確認
def print_route(file_path):
    print("Updated route content:")
    try:
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                print(row)
    except Exception as e:
        print("Failed to print route:", str(e))

# 新しいファイルに更新後の経路を保存
def save_updated_route(route, file_path):
    try:
        with open(file_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y'])  # ヘッダー
            for x, y in route:
                writer.writerow(["{:.2f}".format(x), "{:.2f}".format(y)])  # 小数点以下2桁で保存
        print("Updated route successfully saved to", file_path)
    except Exception as e:
        print("Failed to save updated route:", str(e))


# 経路生成
path = a_star(start, goal, save_path)
if path:
    round_trip_path = create_round_trip_path(path)
    print("Route found and round trip path generated.")
else:
    print("No path found.")


# 既存の経路ファイルを読み込む
#original_route_file = 'saved_route_new.csv'
original_route_file = save_path
print("test")
route = load_route(original_route_file)
print("test")

if route:
    # 反復経路を作成
    round_trip_route = create_round_trip_path(route)

    # 反復経路で既存のCSVファイルを更新
    save_round_trip_route(round_trip_route, original_route_file)

    # 更新後のファイル内容を確認のため再度読み込み
    updated_route = load_route(original_route_file)
    
    # 更新された経路を出力して確認
    print_route(original_route_file)
    
    # 更新された経路を新しいファイルに保存
    new_route_file = ROUTE_FILE_PATH
    save_updated_route(updated_route, new_route_file)
    print("save route:", ROUTE_FILE_PATH)

else:
    print("No valid route to process.")



#csv_look_ahead.py
LOOK_AHEAD_DIST = 1.3  # look-ahead distance [meter]
SPACING = 0.2       # distance between lines 
x_tolerance = 0.1  # [meter]
YAW_TOLERANCE = 50.0 # [Degree]

YAW_TOLERANCE_ONSTART = 5.0 # [Degree]

I_CONTROL_DIST = 0.1 # [meter], refer to cross_track_error 
MAX_PIVOT_COUNT = 1

# translation value
FORWARD_CONST = 1
BACKWARD_CONST = -1

# AJK
TRANSLATION_NEUTRAL = 512     # neutral value
STEERING_NEUTRAL = 512        # neutral value
RIGHT_PIVOT = 332
LEFT_PIVOT = 692
FB_OPTIMUM = 220
LR_OPTIMUM = 60

# for simulator or test vehicle
CMD_LINEAR_OPT = 0.3
CMD_ANGULAR_RIGHT = -0.3
CMD_ANGULAR_LEFT = 0.3
CMD_ANGULAR_K = 0.3
CMD_ANGULAR_LIMIT = 0.3

# gain
KP = 0.03
KI = 1.0
KD = 0.1

# frequency [Hz]
FREQUENCY = 10

# CSVファイルからウェイポイントを読み込む関数を追加 added part
def load_generated_route(file_path):
    waypoint_x = []
    waypoint_y = []
    try:
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # ヘッダーをスキップ
            for row in reader:
                waypoint_x.append(float(row[0]))  # x座標
                waypoint_y.append(float(row[1]))  # y座標
        print("Generated route successfully loaded from", file_path)
    except Exception as e:
        print("Failed to load generated route:", str(e))
    return waypoint_x, waypoint_y

#経路追従部分のコード
class look_ahead():
    def __init__(self):
        # ウェイポイントや現在の状態を初期化
        self.waypoint_x = []  # ウェイポイントのX座標リスト
        self.waypoint_y = []  # ウェイポイントのY座標リスト
        self.waypoint_goal = []  # ゴールウェイポイントリスト
        self.x = 0  # 現在のX座標
        self.y = 0  # 現在のY座標
        self.yaw = np.pi/2  # 現在の向き (初期値は北向き)
        self.pre_steering_ang = 0  # 以前のステアリング角度

        rospy.on_shutdown(self.shutdown)  # 終了時のシャットダウン処理を登録

        # TF情報の取得に必要な設定
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

        # ロボットの速度指令をパブリッシュするための設定
        self.cmdvel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size = 1)
        self.cmdvel = Twist()

        # Lookaheadのログをパブリッシュするための設定
        self.lookahead_log_pub = rospy.Publisher('/lookahead_log', Lookahead_Log, queue_size = 1)
        self.lookahead_log = Lookahead_Log()

        # Lookaheadの状態をパブリッシュするための設定
        self.status_pub = rospy.Publisher('/lookahead/status', String, queue_size = 1)
        self.statusMsg = String()

        # 経路生成で生成されたCSVファイルからウェイポイントを読み込む
        file_path = ROUTE_FILE_PATH  # 経路生成プログラムで生成されたファイルのパス
        self.waypoint_x, self.waypoint_y = load_generated_route(file_path)

        # ウェイポイントの読み込みに失敗した場合はエラーログを表示
        if not self.waypoint_x or not self.waypoint_y:
            rospy.logerr("Failed to load waypoints from the generated route.")

    def cmdvel_publisher(self, steering_ang, translation, pi):
        # ステアリング角が許容値を超える場合の処理
        if abs(steering_ang) > YAW_TOLERANCE:
            # ステアリング角が正の値の場合は左回転
            if steering_ang >= 0:
                self.cmdvel.linear.x = 0  # 前進速度をゼロに設定
                self.cmdvel.angular.z = CMD_ANGULAR_LEFT  # 左回転の角速度を設定
            # ステアリング角が負の値の場合は右回転
            else:
                self.cmdvel.linear.x = 0  # 前進速度をゼロに設定
                self.cmdvel.angular.z = CMD_ANGULAR_RIGHT  # 右回転の角速度を設定
        else:
            # ステアリング角が許容値以内の場合は直進または曲線移動
            self.cmdvel.linear.x = CMD_LINEAR_OPT * translation  # 前進速度を設定
            self.cmdvel.angular.z = pi * CMD_ANGULAR_K  # 曲がり角度に基づく角速度を設定

            # 角速度の制限処理
            if self.cmdvel.angular.z > CMD_ANGULAR_LIMIT:
                self.cmdvel.angular.z = CMD_ANGULAR_LIMIT  # 角速度の上限を設定
            elif self.cmdvel.angular.z < -CMD_ANGULAR_LIMIT:
                self.cmdvel.angular.z = -CMD_ANGULAR_LIMIT  # 角速度の下限を設定

        # 設定した速度指令をパブリッシュ
        self.cmdvel_pub.publish(self.cmdvel)
        return self.cmdvel.linear.x, self.cmdvel.angular.z  # 前進速度と角速度を返す


    def shutdown(self):
        print "shutdown" #途中で中断する場合の命令を設定する

    def loop(self):
        seq = 1  # ウェイポイントの現在のインデックス
        self.last_steering_ang = 0  # 前回のステアリング角度を初期化

        rate = rospy.Rate(10.0)  # 10Hzのループレートを設定
        while not rospy.is_shutdown():
            try:
                # 現在のロボットの位置と姿勢を取得
                t = self.tfBuffer.lookup_transform('map', 'base_link', rospy.Time())
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                print(e)  # エラーを表示
                rate.sleep()
                continue

            # ミッションパラメータの取得 (停止命令がある場合の処理)
            missionParam = rospy.get_param('mission_param', 'start')
            if missionParam != 'start':
                self.cmdvel.linear.x = 0  # 前進速度をゼロに設定
                self.cmdvel.angular.z = 0  # 角速度をゼロに設定
                self.cmdvel_pub.publish(self.cmdvel)  # 停止命令を送信
                rospy.loginfo("stop by rosparam")  # 停止をログに記録
                rate.sleep()
                continue

            # ロボットの現在の位置と姿勢を取得
            rover_pos = t.transform.translation
            rover_quat = t.transform.rotation

            # ウェイポイント間のベクトル計算
            if seq == 0:
                wp_x_adj = self.waypoint_x[seq] - rover_pos.x
                wp_y_adj = self.waypoint_y[seq] - rover_pos.y
                own_x_adj = 0
                own_y_adj = 0
            else:
                wp_x_adj = self.waypoint_x[seq] - self.waypoint_x[seq-1]
                wp_y_adj = self.waypoint_y[seq] - self.waypoint_y[seq-1]
                own_x_adj = rover_pos.x - self.waypoint_x[seq-1]
                own_y_adj = rover_pos.y - self.waypoint_y[seq-1]

            # 座標変換 (ウェイポイントと自己位置を目標ベクトルに合わせる)
            tf_angle = np.arctan2(wp_y_adj, wp_x_adj)
            wp_x_tf = wp_x_adj*np.cos(-tf_angle) - wp_y_adj*np.sin(-tf_angle)
            wp_y_tf = wp_x_adj*np.sin(-tf_angle) + wp_y_adj*np.cos(-tf_angle)

            own_x_tf = own_x_adj*np.cos(-tf_angle) - own_y_adj*np.sin(-tf_angle)
            own_y_tf = own_x_adj*np.sin(-tf_angle) + own_y_adj*np.cos(-tf_angle)


            tf_q = quaternion_from_euler(0, 0, tf_angle)
            front_q_tf = quaternion_multiply((rover_quat.x, rover_quat.y, rover_quat.z, rover_quat.w), 
                                             (     tf_q[0],      tf_q[1],      tf_q[2],     -tf_q[3]))

            rear_q_tf = np.empty(4)
            rear_q_tf[0] = front_q_tf[0]
            rear_q_tf[1] = front_q_tf[1]
            rear_q_tf[2] = front_q_tf[3]
            rear_q_tf[3] = -front_q_tf[2]

            # ステアリング角度と速度制御の計算
            bearing = np.arctan2(-own_y_tf, LOOK_AHEAD_DIST)
            bearing_q = quaternion_from_euler(0, 0, bearing)

            front_steering_q = quaternion_multiply(( bearing_q[0],  bearing_q[1],  bearing_q[2],  bearing_q[3]),
                                                   (front_q_tf[0], front_q_tf[1], front_q_tf[2], -front_q_tf[3]))
            rear_steering_q = quaternion_multiply((bearing_q[0], bearing_q[1], bearing_q[2], bearing_q[3]),
                                                  (rear_q_tf[0], rear_q_tf[1], rear_q_tf[2], -rear_q_tf[3]))

            front_steering_ang = euler_from_quaternion(front_steering_q)[2]/np.pi *180
            rear_steering_ang = euler_from_quaternion(rear_steering_q)[2]/np.pi *180

            
            if abs(front_steering_ang) >= abs(rear_steering_ang):
                steering_ang = rear_steering_ang
                translation = BACKWARD_CONST
            elif abs(front_steering_ang) < abs(rear_steering_ang):
                steering_ang = front_steering_ang
                translation = FORWARD_CONST
            steering_ang = front_steering_ang
            translation = FORWARD_CONST

            # PID制御で角速度を計算
            p = KP *steering_ang
            i = KI *own_y_tf
            d = KD * (self.last_steering_ang - steering_ang)
            self.last_steering_ang = steering_ang

            pi_value = p - d
            if abs(own_y_tf) < I_CONTROL_DIST:
                pi_value = p - i

            # 指令を送信し、結果を記録
            linear_x, angular_z = self.cmdvel_publisher(steering_ang, translation, pi_value)


            yaw = euler_from_quaternion((rover_quat.x, rover_quat.y, rover_quat.z, rover_quat.w))[2]/np.pi * 180.0

            # ログデータを保存
            self.lookahead_log.stamp = rospy.Time.now()
            self.lookahead_log.waypoint_seq = seq
            self.lookahead_log.waypoint_start_x = self.waypoint_x[seq-1]
            self.lookahead_log.waypoint_start_y = self.waypoint_y[seq-1]
            self.lookahead_log.waypoint_end_x = self.waypoint_x[seq]
            self.lookahead_log.waypoint_end_y = self.waypoint_y[seq]
            self.lookahead_log.own_x = rover_pos.x
            self.lookahead_log.own_y = rover_pos.y
            self.lookahead_log.own_yaw = euler_from_quaternion((rover_quat.x, 
                                                                rover_quat.y,
                                                                rover_quat.z,
                                                                rover_quat.w))[2]/np.pi * 180.0
            self.lookahead_log.tf_waypoint_x = wp_x_tf
            self.lookahead_log.tf_waypoint_y = wp_y_tf
            self.lookahead_log.tf_own_x = own_x_tf
            self.lookahead_log.tf_own_y = own_y_tf
            self.lookahead_log.cross_track_error = -own_y_tf
            self.lookahead_log.Kp = KP
            self.lookahead_log.Ki = KI
            self.lookahead_log.Kd = KD
            self.lookahead_log.look_ahead_dist = LOOK_AHEAD_DIST
            self.lookahead_log.i_control_dist = I_CONTROL_DIST
            self.lookahead_log.p = p
            self.lookahead_log.i = i
            self.lookahead_log.d = d
            self.lookahead_log.steering_ang = steering_ang
            self.lookahead_log.linear_x = linear_x
            self.lookahead_log.angular_z = angular_z
            self.lookahead_log_pub.publish(self.lookahead_log)


            self.f = open(LOG_FILE_PATH, 'a')
            csvWriter = csv.writer(self.f)

            csvWriter.writerow([rospy.Time.now().to_sec(), 
                                seq,
                                self.waypoint_x[seq-1],
                                self.waypoint_y[seq-1],
                                self.waypoint_x[seq],
                                self.waypoint_y[seq],
                                rover_pos.x,
                                rover_pos.y,
                                yaw,
                                wp_x_tf,
                                wp_y_tf,
                                own_x_tf,
                                own_y_tf
                                -own_y_tf,
                                KP,
                                KI,
                                KD,
                                LOOK_AHEAD_DIST,
                                I_CONTROL_DIST,
                                p,
                                i,
                                d,
                                steering_ang,
                                linear_x,
                                angular_z])
            self.f.close()

            u = np.array([wp_x_tf, wp_y_tf])
            v = np.array([own_x_tf, own_y_tf])
            d = np.cross(u, v) / (np.linalg.norm(u) + 0.0000001)

            #print("sequence:", seq)
            #print("transform_wx:", wp_x_tf, "transform_wy:", wp_y_tf)
            #print("transform_own_x:", own_x_tf, "transform_own_y:", own_y_tf)
            #print("target_dist:", d)
            #print("steering_angle:",steering_ang)
            #print("linear:",linear_x, "angular:",angular_z)
            #print("\n")
            #print "cross_track_error:", d
            #print front_steering_ang, rear_steering_ang
            #print steering_ang, pd_value
            #print self.ajk_value.translation, self.ajk_value.steering

            # ウェイポイントの進行管理
            if (wp_x_tf - own_x_tf) < x_tolerance:
                pre_wp_x = self.waypoint_x[seq]
                pre_wp_y = self.waypoint_y[seq]
                seq = seq + 1
                try:
                    a = np.array([pre_wp_x, pre_wp_y])
                    b = np.array([self.waypoint_x[seq], self.waypoint_y[seq]])

                    if np.linalg.norm(a-b) < SPACING:
                        seq = seq + 1
                except IndexError:
                    pass

            # すべてのウェイポイントが終了した場合の処理
            if seq >= len(self.waypoint_x):
                self.cmdvel.linear.x = 0
                self.cmdvel.angular.z = 0
                self.cmdvel_pub.publish(self.cmdvel)
                seq = 1
                print "mission_end"

                isNodeAlive = rosnode.rosnode_ping("/mybag", max_count=1, verbose=False)
                if isNodeAlive == True:
                    p=subprocess.Popen(["rosnode", "kill", "/mybag"])
                break
            #print
            rate.sleep() # 次のループまで待機

    def load_waypoint(self):
        # ウェイポイントをCSVファイルから読み込む
        self.waypoint_x, self.waypoint_y = load_waypoint.load_csv()
    
if __name__ == '__main__':
    l = look_ahead() # look_aheadクラスのインスタンスを作成
    #l.load_waypoint()
    l.loop() # ループ処理を開始


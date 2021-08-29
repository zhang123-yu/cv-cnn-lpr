import cv2
import os
import numpy as np
import tensorflow as tf

char_table = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
              'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '川', '鄂', '赣', '甘', '贵',
              '桂', '黑', '沪', '冀', '津', '京', '吉', '辽', '鲁', '蒙', '闽', '宁', '青', '琼', '陕', '苏', '晋',
              '皖', '湘', '新', '豫', '渝', '粤', '云', '藏', '浙']


# 调整输入图片的大小
def img_resize(orig_img):
    img_h = orig_img.shape[0]
    img_w = orig_img.shape[1]

    rate = img_w / img_h  # 长宽比
    re_w = 640  # 宽设为640像素
    re_h = int(re_w / rate)

    if img_w > re_w:
        resized_img = cv2.resize(img, (re_w, re_h), cv2.INTER_AREA)
    else:
        resized_img = orig_img

    return resized_img


# 图片预处理，形态学操作
def pre_process(orig_img):
    gray_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray_img', gray_img)

    blur_img = cv2.blur(gray_img, (3, 3))
    # cv2.imshow('blur', blur_img)

    sobel_img = cv2.Sobel(blur_img, cv2.CV_16S, 1, 0, ksize=3)
    sobel_img = cv2.convertScaleAbs(sobel_img)
    # cv2.imshow('sobel', sobel_img)

    hsv_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2HSV)

    h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]
    # 黄色色调区间[26，34],蓝色色调区间:[100,124]
    blue_img = (((h > 26) & (h < 34)) | ((h > 100) & (h < 124))) & (s > 70) & (v > 70)
    blue_img = blue_img.astype('float32')

    mix_img = np.multiply(sobel_img, blue_img)
    # cv2.imshow('mix', mix_img)

    mix_img = mix_img.astype(np.uint8)

    retn, binary_img = cv2.threshold(mix_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # cv2.imshow('binary',binary_img)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5))
    close_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow('close', close_img)

    return close_img


# 倾斜角度矫正
def img_transform(car_rect, image):
    img_h, img_w = image.shape[:2]
    rect_w, rect_h = car_rect[1][0], car_rect[1][1]
    angle = car_rect[2]
    car_img = []

    return_flag = False
    if car_rect[2] == 0:
        return_flag = True
    if car_rect[2] == -90 and rect_w < rect_h:
        rect_w, rect_h = rect_h, rect_w
        return_flag = True
    if return_flag:
        car_img = image[int(car_rect[0][1] - rect_h / 2):int(car_rect[0][1] + rect_h / 2),
                  int(car_rect[0][0] - rect_w / 2):int(car_rect[0][0] + rect_w / 2)]
        return car_img

    car_rect = (car_rect[0], (rect_w, rect_h), angle)
    box = cv2.boxPoints(car_rect)

    heigth_point = right_point = [0, 0]
    left_point = low_point = [car_rect[0][0], car_rect[0][1]]
    for point in box:
        if left_point[0] > point[0]:
            left_point = point
        if low_point[1] > point[1]:
            low_point = point
        if heigth_point[1] < point[1]:
            heigth_point = point
        if right_point[0] < point[0]:
            right_point = point

    if left_point[1] <= right_point[1]:  # 正角度
        new_right_point = [right_point[0], heigth_point[1]]
        pts1 = np.float32([left_point, heigth_point, right_point])
        pts2 = np.float32([left_point, heigth_point, new_right_point])  # 字符只是高度需要改变
        m = cv2.getAffineTransform(pts1, pts2)
        dst = cv2.warpAffine(image, m, (round(img_w * 2), round(img_h * 2)))
        car_img = dst[int(left_point[1]):int(heigth_point[1]), int(left_point[0]):int(new_right_point[0])]

    elif left_point[1] > right_point[1]:  # 负角度
        new_left_point = [left_point[0], heigth_point[1]]
        pts1 = np.float32([left_point, heigth_point, right_point])
        pts2 = np.float32([new_left_point, heigth_point, right_point])  # 字符只是高度需要改变
        m = cv2.getAffineTransform(pts1, pts2)
        dst = cv2.warpAffine(image, m, (round(img_w * 2), round(img_h * 2)))
        car_img = dst[int(right_point[1]):int(heigth_point[1]), int(new_left_point[0]):int(right_point[0])]

    return car_img


# 根据长宽比例和倾斜角度筛选车牌
def verify_scale(rotate_rect):
    error = 0.4
    aspect = 4.7272
    min_area = 10 * (10 * aspect)
    max_area = 150 * (150 * aspect)
    min_aspect = aspect * (1 - error)
    max_aspect = aspect * (1 + error)
    theta = 30

    # 宽或高为0，不满足矩形直接返回False
    if rotate_rect[1][0] == 0 or rotate_rect[1][1] == 0:
        return False

    r = rotate_rect[1][0] / rotate_rect[1][1]
    r = max(r, 1 / r)
    area = rotate_rect[1][0] * rotate_rect[1][1]
    if min_area < area < max_area and min_aspect < r < max_aspect:
        # 矩形的倾斜角度在不超过theta
        if ((rotate_rect[1][0] < rotate_rect[1][1] and -90 <= rotate_rect[2] < -(90 - theta)) or
                (rotate_rect[1][1] < rotate_rect[1][0] and -theta < rotate_rect[2] <= 0)):
            return True
    return False


# 给候选车牌区域做漫水填充算法，一方面补全上一步求轮廓可能存在轮廓歪曲的问题，另一方面也可以将非车牌区排除掉
def verify_color(rotate_rect, src_image):
    img_h, img_w = src_image.shape[:2]
    mask = np.zeros(shape=[img_h + 2, img_w + 2], dtype=np.uint8)
    connectivity = 4  # 种子点上下左右4邻域与种子颜色值在[lo_diff,up_diff]的被涂成new_value，也可设置8邻域
    lo_diff, up_diff = 30, 30
    new_value = 255
    flags = connectivity
    flags |= cv2.FLOODFILL_FIXED_RANGE  # 考虑当前像素与种子象素之间的差，不设置的话则和邻域像素比较
    flags |= new_value << 8
    flags |= cv2.FLOODFILL_MASK_ONLY  # 设置这个标识符则不会去填充改变原始图像，而是去填充掩模图像（mask）

    rand_seed_num = 5000  # 生成多个随机种子
    valid_seed_num = 200  # 从rand_seed_num中随机挑选valid_seed_num个有效种子
    adjust_param = 0.1
    box_points = cv2.boxPoints(rotate_rect)
    box_points_x = [n[0] for n in box_points]
    box_points_x.sort(reverse=False)
    adjust_x = int((box_points_x[2] - box_points_x[1]) * adjust_param)
    col_range = [box_points_x[1] + adjust_x, box_points_x[2] - adjust_x]
    box_points_y = [n[1] for n in box_points]
    box_points_y.sort(reverse=False)
    adjust_y = int((box_points_y[2] - box_points_y[1]) * adjust_param)
    row_range = [box_points_y[1] + adjust_y, box_points_y[2] - adjust_y]
    # 如果以上方法种子点在水平或垂直方向可移动的范围很小，则采用旋转矩阵对角线来设置随机种子点
    if (col_range[1] - col_range[0]) / (box_points_x[3] - box_points_x[0]) < 0.4 \
            or (row_range[1] - row_range[0]) / (box_points_y[3] - box_points_y[0]) < 0.4:
        points_row = []
        points_col = []
        for i in range(2):
            pt1, pt2 = box_points[i], box_points[i + 2]
            x_adjust, y_adjust = int(adjust_param * (abs(pt1[0] - pt2[0]))), int(adjust_param * (abs(pt1[1] - pt2[1])))
            if pt1[0] <= pt2[0]:
                pt1[0], pt2[0] = pt1[0] + x_adjust, pt2[0] - x_adjust
            else:
                pt1[0], pt2[0] = pt1[0] - x_adjust, pt2[0] + x_adjust
            if pt1[1] <= pt2[1]:
                pt1[1], pt2[1] = pt1[1] + adjust_y, pt2[1] - adjust_y
            else:
                pt1[1], pt2[1] = pt1[1] - y_adjust, pt2[1] + y_adjust
            temp_list_x = [int(x) for x in np.linspace(pt1[0], pt2[0], int(rand_seed_num / 2))]
            temp_list_y = [int(y) for y in np.linspace(pt1[1], pt2[1], int(rand_seed_num / 2))]
            points_col.extend(temp_list_x)
            points_row.extend(temp_list_y)
    else:
        points_row = np.random.randint(row_range[0], row_range[1], size=rand_seed_num)
        points_col = np.linspace(col_range[0], col_range[1], num=rand_seed_num).astype(np.int)

    points_row = np.array(points_row)
    points_col = np.array(points_col)
    hsv_img = cv2.cvtColor(src_image, cv2.COLOR_BGR2HSV)
    h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]
    # 将随机生成的多个种子依次做漫水填充,理想情况是整个车牌被填充
    flood_img = src_image.copy()
    seed_cnt = 0
    for i in range(rand_seed_num):
        rand_index = np.random.choice(rand_seed_num, 1, replace=False)
        row, col = points_row[rand_index], points_col[rand_index]
        # 限制随机种子必须是车牌背景色
        if (((h[row, col] > 26) & (h[row, col] < 34)) | ((h[row, col] > 100) & (h[row, col] < 124))) & (
                s[row, col] > 70) & (v[row, col] > 70):
            cv2.floodFill(src_image, mask, (col, row), (255, 255, 255), (lo_diff,) * 3, (up_diff,) * 3, flags)
            cv2.circle(flood_img, center=(col, row), radius=2, color=(0, 0, 255), thickness=2)
            seed_cnt += 1
            if seed_cnt >= valid_seed_num:
                break
    # 获取掩模上被填充点的像素点，并求点集的最小外接矩形
    mask_points = []
    for row in range(1, img_h + 1):
        for col in range(1, img_w + 1):
            if mask[row, col] != 0:
                mask_points.append((col - 1, row - 1))
    mask_rotate_rect = cv2.minAreaRect(np.array(mask_points))
    if verify_scale(mask_rotate_rect):
        return True, mask_rotate_rect
    else:
        return False, mask_rotate_rect


# 车牌定位
def locate_plate(orig_img, pred_image):
    locate_plate_list = []
    temp1_orig_img = orig_img.copy()  # 调试用
    temp2_orig_img = orig_img.copy()  # 调试用
    clone, contours, heriachy = cv2.findContours(pred_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        cv2.drawContours(temp1_orig_img, contours, i, (0, 255, 255), 2)
        # 获取轮廓最小外接矩形，返回值rotate_rect
        rotate_rect = cv2.minAreaRect(contour)
        # 根据矩形面积大小和长宽比判断是否是车牌
        if verify_scale(rotate_rect):
            retn, rotate_rect2 = verify_color(rotate_rect, temp2_orig_img)
            if not retn:
                continue
            # 车牌位置矫正
            located_plate = img_transform(rotate_rect2, temp2_orig_img)
            located_plate = cv2.resize(located_plate, (car_plate_w, car_plate_h))  # 调整尺寸为后面CNN车牌识别做准备
            locate_plate_list.append(located_plate)

    cv2.imshow('contour', temp1_orig_img)
    cv2.imwrite('./data/output/plate_locate.jpg', temp1_orig_img)

    return locate_plate_list


# 左右切割
def horizontal_cut_chars(plate):
    char_addr_list = []
    area_left, area_right, char_left, char_right = 0, 0, 0, 0
    img_w = plate.shape[1]

    # 获取车牌每列边缘像素点个数
    def get_col_sum(col_sum_img, col_point):
        sum_col = 0
        for i_col in range(col_sum_img.shape[0]):
            sum_col += round(col_sum_img[i_col, col_point] / 255)
        return sum_col

    col_sum = 0
    for col in range(img_w):
        col_sum += get_col_sum(plate, col)
    # 根据每列边缘像素点判断字符区域
    col_limit = round(0.1 * col_sum / img_w)
    # 每个字符宽度也进行限制
    char_wid_limit = [round(img_w / 12), round(img_w / 5)]
    is_char_flag = False

    for i in range(img_w):
        col_value = get_col_sum(plate, i)
        if col_value > col_limit:
            if not is_char_flag:
                area_right = round((i + char_right) / 2)
                area_width = area_right - area_left
                char_width = char_right - char_left
                if (area_width > char_wid_limit[0]) and (area_width < char_wid_limit[1]):
                    char_addr_list.append((area_left, area_right, char_width))
                char_left = i
                area_left = round((char_left + char_right) / 2)
                is_char_flag = True
        else:
            if is_char_flag:
                char_right = i - 1
                is_char_flag = False
    # 手动结束最后未完成的字符分割
    if area_right < char_left:
        area_right, char_right = img_w, img_w
        area_width = area_right - area_left
        char_width = char_right - char_left
        if (area_width > char_wid_limit[0]) and (area_width < char_wid_limit[1]):
            char_addr_list.append((area_left, area_right, char_width))

    if 0 < len(char_addr_list) < 7:
        area_width = char_addr_list[0][1] - char_addr_list[0][0]
        if char_addr_list[0][0] - round(1 * area_width) >= 0:
            char_addr_list.insert(0, (
                char_addr_list[0][0] - round(1 * area_width), char_addr_list[0][1] - round(1 * area_width), area_width))
        else:
            char_addr_list.insert(0, (0, char_addr_list[0][1] - round(1 * area_width), area_width))

    return char_addr_list


# 提取分割后的字符图片
def get_chars(car_plate_char):
    img_h, img_w = car_plate_char.shape[:2]
    h_proj_list = []  # 水平投影长度列表
    h_temp_len, v_temp_len = 0, 0
    h_start_index, h_end_index = 0, 0  # 水平投影记索引
    h_proj_limit = [0.1, 0.9]  # 车牌在水平方向得轮廓长度少于10%或多余90%过滤掉
    char_imgs = []

    # 将二值化的车牌水平投影到Y轴，计算投影后的连续长度，连续投影长度可能不止一段
    h_count = [0 for _ in range(img_h)]
    for row in range(img_h):
        temp_cnt = 0
        for col in range(img_w):
            if car_plate_char[row, col] == 255:
                temp_cnt += 1
        h_count[row] = temp_cnt
        if temp_cnt / img_w < h_proj_limit[0] or temp_cnt / img_w > h_proj_limit[1]:
            if h_temp_len != 0:
                h_end_index = row - 1
                h_proj_list.append((h_start_index, h_end_index))
                h_temp_len = 0
            continue
        if temp_cnt > 0:
            if h_temp_len == 0:
                h_start_index = row
                h_temp_len = 1
            else:
                h_temp_len += 1
        else:
            if h_temp_len > 0:
                h_end_index = row - 1
                h_proj_list.append((h_start_index, h_end_index))
                h_temp_len = 0

    # 手动结束最后得水平投影长度累加
    if h_temp_len != 0:
        h_end_index = img_h - 1
        h_proj_list.append((h_start_index, h_end_index))
    # 选出最长的投影，该投影长度占整个截取车牌高度的比值必须大于0.5
    h_max_index, h_max_height = 0, 0
    for i, (start, end) in enumerate(h_proj_list):
        if h_max_height < (end - start):
            h_max_height = (end - start)
            h_max_index = i
    if h_max_height / img_h < 0.5:
        return char_imgs
    chars_top, chars_bottom = h_proj_list[h_max_index][0], h_proj_list[h_max_index][1]

    plates = car_plate_char[chars_top:chars_bottom + 1, :]
    # cv2.imwrite('./data/output/car.jpg', car_plate_char)
    cv2.imwrite('./data/output/plate_bin.jpg', plates)
    char_addr_list = horizontal_cut_chars(plates)

    for i, addr in enumerate(char_addr_list):
        char_img = car_plate_char[chars_top:chars_bottom + 1, addr[0]:addr[1]]
        char_img = cv2.copyMakeBorder(char_img, 0, 0, 3, 3, cv2.BORDER_CONSTANT, 0)
        char_img = cv2.resize(char_img, (char_w, char_h))
        cv2.imshow('char_img_' + str(i + 1), char_img)
        cv2.imwrite('./data/output/char_' + str(i + 1) + '.jpg', char_img)
        char_imgs.append(char_img)
    return char_imgs


# 识别字符并排列
def extract_char(car_plate_extract):
    gray_plate = cv2.cvtColor(car_plate_extract, cv2.COLOR_BGR2GRAY)
    # retn, binary_plate = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    binary_plate = cv2.adaptiveThreshold(gray_plate, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, -5)
    char_img_lists = get_chars(binary_plate)
    return char_img_lists


# CNN车牌过滤
def cnn_select_plate(plate_list, model_path):
    if len(plate_list) == 0:
        return False, plate_list
    g1 = tf.Graph()
    sess1 = tf.Session(graph=g1)
    with sess1.as_default():
        with sess1.graph.as_default():
            model_dir = os.path.dirname(model_path)
            saver = tf.train.import_meta_graph(model_path)
            saver.restore(sess1, tf.train.latest_checkpoint(model_dir))
            graph = tf.get_default_graph()
            net1_x_place = graph.get_tensor_by_name('x_place:0')
            net1_keep_place = graph.get_tensor_by_name('keep_place:0')
            net1_out = graph.get_tensor_by_name('out_put:0')

            input_x = np.array(plate_list)
            net_outs = tf.nn.softmax(net1_out)
            preds = tf.argmax(net_outs, 1)  # 预测结果
            probs = tf.reduce_max(net_outs, reduction_indices=[1])  # 结果概率值
            pred_list, prob_list = sess1.run([preds, probs], feed_dict={net1_x_place: input_x, net1_keep_place: 1.0})
            # 选出概率最大的车牌
            result_index, result_prob = -1, 0.
            for i, pred in enumerate(pred_list):
                if pred == 1 and prob_list[i] > result_prob:
                    result_index, result_prob = i, prob_list[i]
            if result_index == -1:
                return False, plate_list[0]
            else:
                return True, plate_list[result_index]


# CNN字符识别
def cnn_recognize_char(img_list, model_path):
    g2 = tf.Graph()
    sess2 = tf.Session(graph=g2)
    text_list = []

    if len(img_list) == 0:
        return text_list
    with sess2.as_default():
        with sess2.graph.as_default():
            model_dir = os.path.dirname(model_path)
            saver = tf.train.import_meta_graph(model_path)
            saver.restore(sess2, tf.train.latest_checkpoint(model_dir))
            graph = tf.get_default_graph()
            net2_x_place = graph.get_tensor_by_name('x_place:0')
            net2_keep_place = graph.get_tensor_by_name('keep_place:0')
            net2_out = graph.get_tensor_by_name('out_put:0')

            data = np.array(img_list)
            # 数字、字母、汉字，从67维向量找到概率最大的作为预测结果
            net_out = tf.nn.softmax(net2_out)
            preds = tf.argmax(net_out, 1)
            my_preds = sess2.run(preds, feed_dict={net2_x_place: data, net2_keep_place: 1.0})

            for i in my_preds:
                text_list.append(char_table[i])
            return text_list


if __name__ == '__main__':
    car_plate_w, car_plate_h = 136, 36
    char_w, char_h = 20, 20
    # 训练好的模型路径，名称要核对好
    plate_model_path = os.path.join('./data/model/plate_recognize/model.ckpt-510.meta')
    char_model_path = os.path.join('./data/model/char_recognize/model.ckpt-660.meta')

    img = cv2.imread('./data/testset/set1/1.jpg')

    resize_img = img_resize(img)

    # 预处理
    pred_img = pre_process(resize_img)

    # 车牌定位
    car_plate_list = locate_plate(resize_img, pred_img)

    # CNN车牌过滤
    ret, cnn_plate = cnn_select_plate(car_plate_list, plate_model_path)
    if not ret:
        print("未检测到车牌")
    cv2.imshow('cnn_plate_img', cnn_plate)
    cv2.imwrite('./data/output/plate_cnn.jpg', cnn_plate)

    # 字符提取
    char_img_list = extract_char(cnn_plate)

    # CNN字符识别
    text = cnn_recognize_char(char_img_list, char_model_path)
    print(text)

    cv2.waitKey(0)

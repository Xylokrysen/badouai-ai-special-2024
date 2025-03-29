import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

'''
安装 Python COCO API: pip install pycocotools
'''

def convert_coco_bbox(size, box):
    '''
    Introduction
    --------------------
        计算 box 的长度和原始图像的长度比值
    Parameter
    --------------------
    :param size: 原始图像大小
    :param box: 标注的 box 信息，一个包含边界框信息的元组或列表。
                格式1 [x_min, y_min, x_max, y_max]，x_min, y_min 是框的左上角坐标。x_max, y_max 是框的右下角坐标
                格式2 [x_min, y_min, w, h]，x_min, y_min 是框的左上角坐标。w,h 是box 的宽高
    :return: x, y, w, h 标注 box 和原始图像的比值
    '''
    '''
    dw 和 dh 分别是图像宽度和高度的逆，表示图像每个像素相对于标准化后的比例
    '''
    dw = 1. / size[1]  # dw = 1 / width
    dh = 1. / size[0]  # dh = 1 / height
    x = box[0] + box[2] / 2 - 1  # -1 的操作是为了将边界框坐标调整到相对图像中心的坐标系中（YOLO格式要求中心坐标从0开始）
    y = box[1] + box[3] / 2 - 1
    w = box[2]
    h = box[3]
    x = x * dw
    y = y * dh
    w = w * dw
    h = h * dh
    return x, y, w, h  # 从像素坐标转换为相对于图像宽度和高度的比例


'''
真实目标框（ground truth boxes）的尺寸进行聚类（例如用 K-means 算法）而得到的一组“代表性”框的尺寸。
它们用于与初始生成的 anchor box 对比和调整，使得 anchor box 更符合数据分布，从而提高检测准确性
'''


def box_iou(boxes, clusters):
    '''
    Introduction
    ------------
        计算每个box和聚类中心的iou
    Parameters
    ------------
    :param boxes: 所有的 box 数据。形状 (box_num, 2) 的 numpy 数组，每一行代表一个 box 的 [width, height]。
                  注意：这里的 box 用宽度和高度来描述，且假定所有 box 的左上角都在原点
    :param clusters: 聚类中心。形状 (cluster_num, 2) 的 numpy 数组，每一行代表一个聚类中心的 [width, height]
    :return: iou。形状为 (box_num, cluster_num)，表示每个 box 与每个聚类中心的 IoU 值
    '''

    box_num = boxes.shape[0]
    cluster_num = clusters.shape[0]
    box_area = boxes[:, 0] * boxes[:, 1]  # 将每个 box 的 w 和 h 相乘得到它们的面积，box_area 的形状为 (box_num,)

    # 每个 box 的面积重复 9 次，对应 9 个聚类中心
    box_area = box_area.repeat(cluster_num)  # 将每个 box 的面积复制 cluster_num 次，得到一维数组
    box_area = np.reshape(box_area, [box_num, cluster_num])

    cluster_area = clusters[:, 0] * clusters[:, 1]
    cluster_area = np.tile(cluster_area, [1, box_num])
    cluster_area = np.reshape(cluster_area, [box_num, cluster_num])

    # 这里计算两个矩形的iou，默认所有矩形的左上角坐标都是在原点，然后计算iou，因此只需要取长宽最小值相乘就是重叠区域的面积
    box_width = np.reshape(boxes[:, 0].repeat(cluster_num), [box_num, cluster_num])
    cluster_width = np.reshape(np.tile(clusters[:, 0], [1, box_num]), [box_num, cluster_num])
    min_width = np.minimum(cluster_width, box_width)

    box_high = np.reshape(boxes[:, 1].repeat(cluster_num), [box_num, cluster_num])
    cluster_high = np.reshape(np.tile(clusters[:, 1], [1, box_num]), [box_num, cluster_num])
    min_high = np.minimum(cluster_high, box_high)

    iou = np.multiply(min_high, min_width) / (box_area + cluster_area - np.multiply(min_high, min_width))
    return iou


def avg_iou(boxes, clusters):
    '''
    Introduction
    ------------
        计算所有box和聚类中心的最大iou均值作为准确率
    Parameters
    ----------
    :param boxes: 所有的 box（anchor box 框）
    :param clusters: 聚类中心（通过标签聚类得到的框）
    :return: accuracy: 准确率
    '''
    return np.mean(np.max(box_iou(boxes, clusters), axis=0))


def Kmeans(boxes, cluster_num, iteration_cutoff=25, function=np.median):
    '''
    Introduction
    ------------
        根据所有box的长宽进行Kmeans聚类
    Parameters
    ----------
    :param boxes: 所有的 box 的长宽
    :param cluster_num: 聚类的数量
    :param iteration_ctoff: 当准确率不再降低多少轮次迭代
    :param function: 聚类中心更新的方式
    :return: clusters: 聚类中心 box 的大小
    '''
    boxes_num = boxes.shape[0]
    best_average_iou = 0  # 用于记录迭代中遇到的最佳平均 IoU
    best_avg_iou_iteration = 0  # 记录获得最佳平均 IoU 时的迭代轮数
    best_clusters = 0  # 保存当时的聚类中心
    anchors = []  # 最终返回的 anchor boxes 列表
    np.random.seed()  # 不传参时，使用当前时间随机种子，确保初始聚类中心的随机性

    # 随机选择所有 boxes 中的 box 作为聚类中心
    clusters = boxes[np.random.choice(boxes_num, cluster_num, replace=False)]
    count = 0

    while True:
        distances = 1. - box_iou(boxes, clusters)  # 用 1 - IoU 作为距离（IoU 越大，距离越小）
        boxes_iou = np.min(distances, axis=1)  # 选取每一行的最小值
        current_box_cluster = np.argmin(distances, axis=1)  # 获取每一行中最小值对应的索引
        average_iou = np.mean(1. - boxes_iou)  # 得到所有 box 与其分配聚类中心的平均 IoU 值。该值用来评价当前聚类效果。
        if average_iou > best_average_iou:
            best_average_iou = average_iou
            best_clusters = clusters
            best_avg_iou_iteration = count

        # 通过 function 方式更新聚类中心
        for cluster in range(cluster_num):
            clusters[cluster] = function(boxes[current_box_cluster == cluster], axis=0)  # 分别取长宽对应的中位数来确定新的聚类中心
        if count > best_avg_iou_iteration + iteration_cutoff:
            break
        print('Sum of all distances (cost) = {}'.format(np.sum(boxes_iou)))
        print('iter: {} Accuracy: {:.2f}%'.format(count, avg_iou(boxes, clusters) * 100))
        count += 1

    for cluster in best_clusters:
        anchors.append([round(cluster[0] * 416), round(cluster[1] * 416)])

    return anchors,best_average_iou

def load_cocoDataset(annfile):
    '''
    Introduction
    ------------
        读取coco数据集的标注信息
    Parameters
    ----------
    :param annfile: COCO标注文件的路径（JSON格式）
    :return: 所有目标框的宽度和高度信息（经过转换后，即以比例形式表示的宽高）
    '''

    data = []
    coco = COCO(annfile) # 使用 COCO API 加载标注文件，创建一个 COCO 数据集对象

    '''
    coco.getCatIds()：获取数据集中所有类别（category）的 ID
    coco.loadCats(...)：加载所有类别的详细信息，返回类别字典列表
    coco.loadImgs()：加载数据集中所有图像的信息（虽然这里没有赋值给变量，但在 COCO 对象内部会存储图像信息）
    '''
    cats = coco.loadCats(coco.getCatIds())
    coco.loadImgs()
    base_classes = {cat['id']:cat['name'] for cat in cats}  # 通过字典推导式构造一个映射，键为类别的 ID，值为类别名称。这在后面用来遍历每个类别对应的图像

    imgId_catIds = [coco.getImgIds(catIds=cat_ids) for cat_ids in base_classes.keys()]

    '''
    举例：
    imgId_catIds = [
    [101, 102, 103],  # 类别 1 中的图像 ID
    [201, 202],        # 类别 2 中的图像 ID
    [301, 302, 303]    # 类别 3 中的图像 ID
    ]
    得到：image_ids = [101, 102, 103, 201, 202, 301, 302, 303]
    '''
    image_ids = [img_id for img_cat_id in imgId_catIds for img_id in img_cat_id]  # 列表推导式两层循环

    for image_id in image_ids:
        annIds = coco.getAnnIds(imgIds=image_id)  # 获取该图像的所有标注（annotation）的 ID
        anns = coco.loadAnns(annIds)  # 加载这些标注的详细信息，返回一个标注列表 anns
        img = coco.loadImgs(image_id)[0]  # 加载该图像的详细信息，loadImgs 返回一个列表，取第一个元素（因为传入的 image_id 只对应一张图像）
        image_width = img['width']
        image_height = img['height']

        for ann in anns:
            box = ann['bbox']  # 取出标注的边界框信息。注意 COCO 中边界框的格式通常为 [xmin, ymin, w, h]（但也可能需要转换）
            bb = convert_coco_bbox((image_width,image_height),box)
            data.append(bb[2:])
    return np.array(data)


def process(dataFile,cluster_num,iteration_cutoff = 25,function = np.median):
    '''
    Introduction
    ------------
        主处理函数
    Parameters
    ----------
    :param dataFile: 数据集的标注文件
    :param cluster_num: 聚类中心数目
    :param iteration_cutoff: 当准确率不再降低多少轮次停止迭代
    :param function: 聚类中心更新的方式
    :return:
    '''

    last_best_iou = 0
    last_anchors = []
    boxes = load_cocoDataset(dataFile)
    box_w = boxes[:1000,0]
    box_h = boxes[:1000,1]
    plt.scatter(box_h,box_w,c='r')
    anchors = Kmeans(boxes,cluster_num,iteration_cutoff,function)
    plt.scatter(anchors[:,0],anchors[:,1],c='b')
    plt.show()

    for _ in range(100):
        anchors,best_iou = Kmeans(boxes,cluster_num,iteration_cutoff,function)
        if best_iou > last_best_iou:
            last_anchors = anchors
            last_best_iou = best_iou
            print('anchor: {}, avg_iou: {}'.format(last_anchors,last_best_iou))
    print('final anchors: {},avg_iou: {}'.format(last_anchors,last_best_iou))




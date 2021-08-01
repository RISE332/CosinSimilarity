import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

T = 60
# File Path
df = pd.read_csv('data/selected_result_rss.csv')
sku_id_list = df['sku_id'].unique()


def draw_hist(data: list, fig_name: str, bins=3):
    """
    :param data:
    :param fig_name:
    :param bins: 将求出的相似度分成几个区间统计
    :return:
    """
    plt.hist(data, bins)
    plt.xlabel('Probability')
    plt.ylabel('Quantity')
    plt.title('{} vector similarity'.format(fig_name))
    plt.grid(True)
    plt.savefig("output/hist/" + fig_name + '.png')
    # plt.show()
    plt.close()


def draw_plot(data: list, fig_name):
    plt.plot(data)
    plt.xlabel('Probability')
    plt.ylabel('Quantity')
    plt.title('{} vector similarity'.format(fig_name))
    plt.savefig("output/plot/" + fig_name + '.png')
    # plt.show()
    plt.close()


def str_split_array(history_sales: str) -> np.array:
    return np.array(history_sales.split(',')).astype(np.float32)


def vector_vector(arr, brr):
    # return arr.dot(brr.T) / (np.sqrt(np.sum(arr*arr)) * np.sqrt(np.sum(brr*brr)))
    return np.sum(arr * brr) / (np.sqrt(np.sum(arr * arr)) * np.sqrt(np.sum(brr * brr)))


def cal_similarity(sku_id):
    """向量与向量的余弦相似度
    1、以sku的第一条数据作为基准向量
    2、计算当前sku剩余行向量和基准向量的余弦相似度
    3、输出矩阵
    """
    input_data = df[df['sku_id'] == sku_id]
    result = []
    j = 1
    for i in range(0, len(input_data['hist_sales']) - 1):
        x = str_split_array(input_data['hist_sales'].iloc[i])
        y = str_split_array(input_data['hist_sales'].iloc[j])
        result.append(vector_vector(arr=x, brr=y))
        j += 1
    return result


def run():
    for sku in sku_id_list:
        draw_hist(cal_similarity(sku), sku)
        draw_plot(cal_similarity(sku), sku)
        # print(sku, len(result), '\t')
        print(sku, '.png is saved')


if __name__ == '__main__':
    run()
    print('finished')

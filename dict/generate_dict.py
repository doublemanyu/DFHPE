import numpy as np


def generate_dict():
    path = r"/mnt/data/ynn/mt/dict/"
    names = ["scy_01", "scy_02", "scy_03", "scy_04"]
    train_data = {}
    test_data = {}
    for name in names:
        if name != "scy_04":
            for i in range(2400):
                train_data[name + "_frame{:0>4d}.npy".format(i)] = 20000
        else:
            for i in range(2400):
                test_data[name + "_frame{:0>4d}.npy".format(i)] = 20000
    np.save(path + "Dict_train.npy", train_data)
    np.save(path + "Dict_test.npy", test_data)


def generate_pred():
    path = r"/mnt/data/ynn/mt/dict/"
    for name in ["jcl_", "scy_"]:
        for i in range(1, 11):
            data = {}
            for j in range(2400):
                data[name + "{:0>2d}_frame{:0>4d}.npy".format(i, j)] = 20000
            np.save(path + name + "{:0>2d}_dict.npy".format(i), data)


def view_npy():
    path = r"/mnt/data/ynn/DHP19_our/train/data/mt_06_frame1007.npy"
    data = np.load(path, allow_pickle=True).item()
    for k in data:
        print(data[k])
        break


if __name__ == "__main__":
    # generate_dict()
    # view_npy()
    generate_pred()


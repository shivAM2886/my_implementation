import numpy as np
import pandas as pd


def find_max_1(list_18):
    max_value = max(list_18)
    for i18 in range(len(list_18)):
        if list_18[i18] == max_value:
            return i18 + 1


def work_function(train_x1, train_y1, length1, list1, k1):
    temp_x1 = np.power(np.array([list1 for i1 in range(length1)]) - train_x1, 2)
    sq_distance = temp_x1.sum(axis=1)

    work_array = np.column_stack((sq_distance, train_y1))
    work_array = work_array[work_array[:, 0].argsort()]
    count = [0, 0]

    for i8 in range(k1):
        if work_array[i8, 1] == 0:
            count[0] += 1
        elif work_array[i8, 1] == 1:
            count[1] += 1
    return find_max_1(count)


def best_k(file_path, dist_parameter_num, k_max):

    data = pd.read_csv(file_path)
    print(data.dtypes)
    parameter_arr = []
    print("Enter parameter strings to be used to calculate distance")
    print("___Note: Entered string and file header must be same___")
    for i in range(dist_parameter_num):
        user_input_1 = input("")
        parameter_arr.append(user_input_1)

    print("Enter classifier_string")
    classifier_string = input("")

    temp_1 = np.random.random(len(data)) < 0.8
    train = data[temp_1]
    test = data[~temp_1]

    train_x = np.asanyarray(train[parameter_arr])
    length = len(train_x)
    train_y = np.asanyarray(train[classifier_string])
    train_y.reshape(1, length)

    test_x = np.asanyarray(test[parameter_arr])
    test_y = np.asanyarray(test[classifier_string])
    length_test = len(test_x)
    test_y.reshape(1, length_test)

    accuracy_array = []
    max_percentage_accuracy = 0
    k_best = 0
    for k in range(2, k_max):
        print(k)
        correct_count = 0
        for i5 in range(length_test):
            list_2 = test_x[i5].tolist()
            result = work_function(train_x, train_y, length, list_2, k)
            if result == test_y[i5]:
                correct_count = correct_count + 1
        percentage_accuracy = (correct_count * 100) / length_test
        print("percentage_accuracy", percentage_accuracy)
        if percentage_accuracy > max_percentage_accuracy:
            max_percentage_accuracy = percentage_accuracy
            k_best = k
        accuracy_array.append(percentage_accuracy)

    return k_best, accuracy_array, train_x, train_y


def k_nearest_neighbour(source_file_path, source_dist_parameter_num, source_k_max, source_pred_list):
    best_k_parameter, array_accuracy, train_xx, train_yy = best_k(source_file_path, source_dist_parameter_num, source_k_max)

    return work_function(train_xx, train_yy, len(train_xx), source_pred_list, best_k_parameter)


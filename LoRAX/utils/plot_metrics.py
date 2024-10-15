"""
Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
SPDX-License-Identifier: MIT
"""
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import yaml


def calc_average_accuracy(accuracy_matrix, column):
    avg_accuracies = np.mean(accuracy_matrix, axis=0)
    return avg_accuracies[column]

# @param log_path: given the log file acc_per_task_remapped
# returns: nxn matrix (where there are n tasks) for tasks t_0 through t_n and each entry R_(i, j), 
# represents the accuracy of classification for task i after task j
def construct_matrix(data):
    acc_per_task_remapped = []

    for i, item  in enumerate(data):
        acc_per_task_remapped.append(item['acc_per_task_remapped'][::-1])
    
        
    for i in range(0, len(acc_per_task_remapped)):
        for j in range(i, len(acc_per_task_remapped)-1):
            acc_per_task_remapped[i].insert(0, 0)
        #print(len(acc_per_task_remapped[i]))
    #print(acc_per_task_remapped)
    acc_per_task_remapped = np.array(acc_per_task_remapped)
    return acc_per_task_remapped
    

# @param accuracy_matrix: an nxn accuracy matrix represented using a numpy array
# bwt equation: (1/(n - 1) * summation {1, i, n - 1} of R_(T, i) - R(i, i))
# accuracy on task i after training on task n - accuracy on task i after training on task i
def calc_bwt(accuracy_matrix):
    sum = 0
    for i in range(0, len(accuracy_matrix) - 1):
        sum += accuracy_matrix[i, len(accuracy_matrix) - 1] - accuracy_matrix[i, i]
    
    return sum/(len(accuracy_matrix) - 1)

# forgetting = -bwt
def calc_forgetting(accuracy_matrix):
    return -calc_bwt(accuracy_matrix)


#converts np matrix to a dataframe with task names and stores it as a string
def matrix_to_string(np_matrix, config_path):
    dict = {}

    #map task indices to task names
    with open (config_path) as config_json:
        #get path to the correct order file
        taskname_path = json.load(config_json)['options'][1]
        # print(taskname_path)
        with open(taskname_path) as tasknames_yaml:
            tasknames = yaml.full_load(tasknames_yaml)['task_name']
            # print(tasknames)
            for i in range(0, len(np_matrix[0])):
                dict[tasknames[i]] = np_matrix[:, i]

    tbl = tbl_df = pd.DataFrame(dict)
    tbl = tbl.set_axis(tasknames[:len(np_matrix[0])], axis='index')
    return tbl.to_string(), tbl_df, tasknames

def get_color_dict():
    colors_i_like = ['blue', 'orange', 'green','red', 'purple', 'navy', 'brown', 'pink', 'gray', 'olive', 'cyan', 'slateblue']
    tasknames = ["gaugan","biggan","cyclegan","imle","deepfake","crn","wild","glow","stargan_gf","stylegan","whichfaceisreal","san"]
    color_dict = dict()
    for i, taskname in enumerate(tasknames):
        color_dict[taskname] = colors_i_like[i]

    return color_dict

# takes in a np accuracy matrix and plots it
def create_lineplot(np_matrix, model, tasknames, scenario, plot_export_path, avg_acc, bwt):
    avg_acc_line = []
    tasks = []
    for i in range(len(np_matrix)):
        avg_acc_line.append(np.mean(np_matrix[0:i+1], axis=0)[i]/100)
        tasks.append(f"Task {i}")
    # print(avg_acc_line)

    color_dict = get_color_dict()
    plt.style.use('bmh')
    plt.title(f'Task Accuracy {model} {scenario} \n aaf: {round(avg_acc, 4)} \n bwt: {round(bwt, 4)}'), 
    plt.xlabel('CIL Class Name'), plt.ylabel('Class Accuracy')

    for i in range(len(np_matrix)):
        y = np.ma.masked_where((np_matrix[i] == 0), np_matrix[i])
        plt.plot(tasks, y/100, marker='s', ms=3.9, label = tasknames[i], color=color_dict[tasknames[i]])

    plt.plot(tasks, avg_acc_line, marker='s', ms=3.9, label='average accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1.05)
    plt.subplots_adjust(bottom=0.2, top=0.8)
    plt.grid(True, which='both', axis='both')
    plt.legend(fontsize='7.2', loc='best')

    #save plot to png file in output directory
    plt.savefig(plot_export_path)

#streamlines entire plotting process given path to logs with accuracy list, path to configs, path to export table to
def process_metrics(task_json_data, config_path, model, scenario, export_path, plot_export_path):
    #first adjust logs json formatting to make it readable
    #then construct the accuracy matrix & rotate
    acc_per_task_remapped = construct_matrix(task_json_data)
    acc_per_task_remapped = np.rot90(acc_per_task_remapped, k=1, axes=(0, 1))
    print(acc_per_task_remapped)

    #calculate various metrics
    metrics_dict = dict()
    metrics_dict['acc_matrix'], tbl_df, tasknames = matrix_to_string(acc_per_task_remapped, config_path)
    metrics_dict['avg_acc_final'] = calc_average_accuracy(acc_per_task_remapped, len(acc_per_task_remapped) - 1)
    metrics_dict['bwt'] = calc_bwt(acc_per_task_remapped)
    metrics_dict['forgetting'] = calc_forgetting(acc_per_task_remapped)
    
    create_lineplot(
        acc_per_task_remapped, 
        model, 
        tasknames, 
        scenario, 
        plot_export_path, 
        metrics_dict['avg_acc_final'], 
        metrics_dict['bwt']
    )

    #export to a plot in the same folder
    with open(export_path, 'w') as export_file:
        export_file.write(str(metrics_dict))


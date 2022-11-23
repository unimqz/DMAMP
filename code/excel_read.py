from openpyxl import load_workbook
import numpy as np
import os
import pickle

def read_name(partition):
  data = []
  for i in partition:
      #print(i)
      content = open(i).readlines()
      for con in content:
          if '>' in con:
              print('AP' + con.strip().replace('>', '').replace(' ', '').split('|')[0])
              data.append(con.strip().replace('>', '').replace(' ', '').split('|')[0])  # APD3 special >000127|human however,the original is >AP000127
              #data.append('AP' + con.strip().replace('>', '').replace(' ', '').split('|')[0])  # APD3 special >000127|human however,the original is >AP000127
      return data

def read_xlsx(wb):
    peptides_infor = {}
    # 获取所有表格(worksheet)的名字
    sheets = wb.get_sheet_names()
    len_sheet = len(sheets)

    for i in range(len_sheet):
        # 第一个表格的名称
        sheet_first = sheets[i]
        # 获取特定的worksheet
        ws = wb.get_sheet_by_name(sheet_first)

        # 获取表格所有行和列，两者都是可迭代的
        rows = ws.rows
        columns = ws.columns

        # 迭代所有的行
        name = []
        for row in rows:
            line = [col.value for col in row][0]
            if line == '':
                break
            name.append(line)
        #print(name)
        peptides_infor[sheet_first] = name
    return peptides_infor

def excel_label(excel_name, data_path, name_file):
    # 打开一个workbook

    wb = load_workbook(filename=excel_name)
    peptide_infor = read_xlsx(wb)

    all_data = read_name(name_file)
    peptide_label = {}
    file_peplist = []
    row_len = 0
    for key_name in peptide_infor.keys():
        content_list = peptide_infor[key_name]
        #print(content_list)
        label = []
        file_peplist.append(key_name)
        for name in all_data:
            name = name.split('|')[0]
            if name in content_list:
                label.append(1)
            else:
                label.append(0)

        peptide_label[key_name] = label
        row_len = len(label) #879

    # print(peptide_label['Antibacterial_2743'])
    one_hot = np.zeros(shape=(row_len, 1))
    for ind, i in enumerate(file_peplist):
        label = peptide_label[i]
        label_np = np.array(label).reshape(len(label), 1)
        # print(label_np)
        # print(i, len(list(np.where(label_np==1))[0])) #统计个数
        if ind == 0:
            one_hot = one_hot + label_np
            #print('one', one_hot.shape)
        else:
            one_hot = np.hstack([one_hot, label_np])
        # print(i, one_hot.shape)

    return one_hot


def camp3_label(name_file):
    # print(name_file)
    infor_list = reafFa(name_file[0])
    seqnamelist,  seqlist = [], []
    lablelist = np.array(np.zeros(shape=(7,)))
    for ind, (seqname, label, seq) in enumerate(infor_list):
        # print(label)
        if ind != 0:
            lablelist= np.vstack((lablelist, label))
        else:
            lablelist = label
            # print('index 0 ', lablelist.shape)
        seqnamelist.append(seqname)
        seqlist.append(seq)
    # print(len(seqnamelist), len(seqlist), len(lablelist))
    # print(seqnamelist[0])
    # print(lablelist[0])
    # print(seqlist[0])
    #
    # # print(lable[0])
    # print(lablelist)
    # exit()
    return seqnamelist, np.array(lablelist), seqlist

def DBAASP_label(name_file):
    # print(name_file)
    infor_list = reafFa(name_file[0])
    seqnamelist,  seqlist = [], []
    lablelist = np.array(np.zeros(shape=(7,)))
    for ind, (seqname, label, seq) in enumerate(infor_list):
        # print(label)
        if ind != 0:
            lablelist= np.vstack((lablelist, label))
        else:
            lablelist = label
            # print('index 0 ', lablelist.shape)
        seqnamelist.append(seqname)
        seqlist.append(seq)
    # print(len(seqnamelist), len(seqlist), len(lablelist))
    # print(seqnamelist[0])
    # print(lablelist[0])
    # print(seqlist[0])
    #
    # # print(lable[0])
    # print(lablelist)
    # exit()
    return seqnamelist, np.array(lablelist), seqlist



def reafFa(fa):
    with open(fa, 'r') as FA:
        seqName, seq = '', ''
        label = np.zeros(shape=(7,))
        while 1:
            line = FA.readline()
            line = line.strip('\n')
            if (line.startswith('>') or not line) and seqName:
                yield((seqName, label, seq))
            if line.startswith('>'):
                seqName = line.split('|')[0].replace('>', '')
                label[1, ] = line.strip().split('|')[-1]
                label[2, ] = line.strip().split('|')[-2]
                label[4, ] = line.strip().split('|')[-3]
                label[3, ] = line.strip().split('|')[-4]
                label[6,] = line.strip().split('|')[-5]

                seq = ''
            else:
                seq += line
            if not line: break
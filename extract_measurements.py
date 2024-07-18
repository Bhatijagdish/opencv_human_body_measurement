

import numpy as np
import sys
import os
import utils

DATA_DIR = "data"
# loading data: file_list, vertex, mean, std
#def obj2npy(label="male"):
#    
#  OBJ_DIR = os.path.join(DATA_DIR, "obj")
#  
#  obj_file_dir = os.path.join(OBJ_DIR, label)
##  print("File directory = ",obj_file_dir)
#  file_list = os.listdir(obj_file_dir)
#
#  # load original data
#  vertex = []
#  for i, obj in enumerate(file_list):
#    sys.stdout.write('\r>> Converting %s body %d\n'%(label, i))
#    sys.stdout.flush()
#    f = open(os.path.join(obj_file_dir, obj), 'r')
#    j = 0
#    for line in f:
#      if line[0] == '#':
#        continue
#      elif "v " in line:
#        line.replace('\n', ' ')
#        tmp = list(map(float, line[1:].split()))
#        vertex.append(tmp)
#        j += 1
#      else:
#        break
# 
#  vertex = np.array(vertex).reshape(len(file_list), utils.V_NUM, 3)#utils.V_NUM
##  print("Vertex are of type = ",type(vertex)) 
##  print("vertex = ",vertex)
#
#  return vertex


        
 
# read control  points(CP) from text file
def convert_cp():
    
  f = open(os.path.join(DATA_DIR, 'customBodyPoints.txt'), "r")

  tmplist = []
  cp = []
  for line in f:
    if '#' in line:
      if len(tmplist) != 0:
        cp.append(tmplist)
        tmplist = []
    elif len(line.split()) == 1:
      continue
    else:
      tmplist.append(list(map(float, line.strip().split())))
  cp.append(tmplist)


  return cp


# calculate measure data from given vertex by control points
def calc_measure(cp, vertex,height):#, facet):
  measure_list = []
  
  for measure in cp:
#    print("#########################",measure)  
#    print("@@@@@@@@@@@@")

    length = 0.0
    p2 = vertex[int(measure[0][1]), :]

    for i in range(0, len(measure)):#1
      p1 = p2
      if measure[i][0] == 1:
        p2 = vertex[int(measure[i][1]), :]  
        
      elif measure[i][0] == 2:
        p2 = vertex[int(measure[i][1]), :] * measure[i][3] + \
        vertex[int(measure[i][2]), :] * measure[i][4]
#        print("if 2 Measurement",int(measure[i][1]))
        
      else:
        p2 = vertex[int(measure[i][1]), :] * measure[i][4] + \
          vertex[int(measure[i][2]), :] * measure[i][5] + \
          vertex[int(measure[i][3]), :] * measure[i][6]
      length += np.sqrt(np.sum((p1 - p2)**2.0))

    measure_list.append(length * 100)# * 1000
  
  measure_list = float(height)*(measure_list/measure_list[0])
#  print("measure list = ",float(height)*(measure_list/measure_list[0])) 
  measure_list[8] = measure_list[8] * 0.36#reducing the error in measurement added due to unarranged vertices
  measure_list[3] = measure_list[3] * 0.6927
#  print("measure list = ",float(height)*(measure_list/measure_list[0]))
#  measure_list = float(height)*(measure_list/measure_list[0])
  return np.array(measure_list).reshape(utils.M_NUM, 1)


##added code: extract body measurements given a .obj model in data.
def extract_measurements(height, vertices, gender, region):
  genders = ["male"]#, "male"]
  measure = []
  for gender in genders:
    # generate and load control point from txt to npy file
    cp = convert_cp()

#    vertex = obj2npy(gender)[0]
    #calculte + convert
    measure = calc_measure(cp, vertices, height)

    face_path = './src/tf_smpl/smpl_faces.npy'
    faces = np.load(face_path)
    obj_mesh_name = 'test.obj'
    with open(obj_mesh_name, 'w') as fp:
        for v in vertices:
            fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )
        for f in faces: # Faces are 1-based, not 0-based in obj files
            fp.write( 'f %d %d %d\n' %  (f[0] + 1, f[1] + 1, f[2] + 1) )

        
    print("Model Saved...")
    
    #give body measurements one by one
    for i in range(0, utils.M_NUM):
      print("%s: %f" % (utils.M_STR[i], measure[i]))
      if gender == "male" and region == "US" and utils.M_STR[i]=="chest":
        if 81 <= (measure[i]) <= 86:
          return ("Dress Size XS")
        elif 86 < (measure[i]) <= 91:
          return ("Dress Size S")
        elif 91 < (measure[i]) <= 97:
          return ("Dress Size M")
        elif 97 < (measure[i]) <= 103:
          return ("Dress Size L")
        elif 103 < (measure[i]) <= 110:
          return ("Dress Size XL")
        elif 110 < (measure[i]) <= 120:
          return ("Dress Size XXL")
      elif gender == "female" and region == "US" and utils.M_STR[i]=="chest":
        if 71 <= (measure[i]) <= 76:
          return ("Dress Size XS")
        elif 76 < (measure[i]) <= 81:
          return ("Dress Size S")
        elif 81 < (measure[i]) <= 86:
          return ("Dress Size M")
        elif 86 < (measure[i]) <= 91:
          return ("Dress Size L")
        elif 91 < (measure[i]) <= 97:
          return ("Dress Size XL")
        elif 97 < (measure[i]) <= 105:
          return ("Dress Size XXL")
      elif gender == "male" and region == "UK" and utils.M_STR[i]=="chest":
        if 86 <= (measure[i]) <= 91:
          return ("Dress Size XS")
        elif 91 < (measure[i]) <= 97:
          return ("Dress Size S")
        elif 97 < (measure[i]) <= 103:
          return ("Dress Size M")
        elif 103 < (measure[i]) <= 110:
          return ("Dress Size L")
        elif 110 < (measure[i]) <= 117:
          return ("Dress Size XL")
        elif 117 < (measure[i]) <= 124:
          return ("Dress Size XXL")
      elif gender == "female" and region == "UK" and utils.M_STR[i]=="chest":
        if 76 <= (measure[i]) <= 81:
          return ("Dress Size XS")
        elif 81 < (measure[i]) <= 87:
          return ("Dress Size S")
        elif 87 < (measure[i]) <= 93:
          return ("Dress Size M")
        elif 93 < (measure[i]) <= 99:
          return ("Dress Size L")
        elif 99 < (measure[i]) <= 105:
          return ("Dress Size XL")
        elif 105 < (measure[i]) <= 111:
          return ("Dress Size XXL")
      elif gender == "male" and region == "EU" and utils.M_STR[i]=="chest":
        if 84 <= (measure[i]) <= 87:
          return ("Dress Size XS")
        elif 87 < (measure[i]) <= 92:
          return ("Dress Size S")
        elif 92 < (measure[i]) <= 97:
          return ("Dress Size M")
        elif 97 < (measure[i]) <= 102:
          return ("Dress Size L")
        elif 102 < (measure[i]) <= 107:
          return ("Dress Size XL")
        elif 107 < (measure[i]) <= 112:
          return ("Dress Size XXL")
      elif gender == "female" and region == "EU" and utils.M_STR[i]=="chest":
        if 76 <= (measure[i]) <= 80:
          return ("Dress Size XS")
        elif 80 < (measure[i]) <= 84:
          return ("Dress Size S")
        elif 84 < (measure[i]) <= 88:
          return ("Dress Size M")
        elif 88 < (measure[i]) <= 92:
          return ("Dress Size L")
        elif 92 < (measure[i]) <= 97:
          return ("Dress Size XL")
        elif 97 < (measure[i]) <= 102:
          return ("Dress Size XXL")
      # else:
      #   return ("Could not determine! Try again!")
    
    
    
    


#if __name__ == "__main__":
#  extract_measurements()
  

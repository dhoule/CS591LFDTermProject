

import csv
import glob

startPath = '/Users/caine2003/Documents/CS591LFDTermProject/initial_output/'

versions = {}

for absPath in glob.glob(startPath + '*.csv'):
  with open(absPath, 'r') as csvFile:
    info = csv.reader(csvFile,delimiter=',')
    fileName = absPath.split('/')[-1].split('.')[0]
    fileInfo = fileName.split('_')
    if fileInfo[1] not in versions:
      versions[fileInfo[1]] = {}

    trip = True
    versions[fileInfo[1]][fileInfo[5]] = []
    for row in info:
      if trip:
        trip = False
        continue
      versions[fileInfo[1]][fileInfo[5]].append(float(i) for i in row)

classified = {}
# Now, need to go through `versions` and determine the best run
  # Look at the last epoch of the training step, the 2nd to last row:
  # The lowest loss value and absolute lowest delta loss value
  # The accuracy needs to be 1.0 in the last epoch of the training step
for version, runs in versions.items():
  current = {}
  current['run'] = None
  current['epochs'] = None
  current['loss'] = -100.0
  current['delta_loss'] = 0.0
  current['delta_acc'] = 0.0
  for run, epochs in runs.items():
    # `epochs` is a Generator object & cannot be accessed like a list. So it needs to be converted.
    temp = []
    for row in epochs:
      temp.append(list(row))

    # If the current run never classified the data, move onto the next one
    if temp[-1][-2] != 1.0 and temp[-2][-2] != 1.0:
      continue
    # This is the first time a run has been seen 
    if current['run'] == None:
      current['run'] = run
      current['epochs'] = temp
      current['loss'] = temp[-2][-1]
      current['delta_loss'] = abs(temp[-1][-1] - temp[-2][-1])
      current['delta_acc'] = abs(temp[-1][-2] - temp[-2][-2])
      continue
    
    if (temp[-2][-1] < current['loss']) and (abs(temp[-1][-1] - temp[-2][-1]) < current['delta_loss']) and (abs(temp[-1][-2] - temp[-2][-2]) < current['delta_acc']):
      current['run'] = run
      current['epochs'] = temp
      current['loss'] = temp[-2][-1]
      current['delta_loss'] = abs(temp[-1][-1] - temp[-2][-1])
      current['delta_acc'] = abs(temp[-1][-2] - temp[-2][-2])
  
  if current['run'] != None:
    if version not in classified:
      classified[version] = {}
      classified[version]['run'] = current['run']
      classified[version]['epochs'] = current['epochs']
      classified[version]['loss'] = current['loss']
      classified[version]['delta_loss'] = current['delta_loss']
      classified[version]['delta_acc'] = current['delta_acc']
    else:
      classified[version]['run'] = current['run']
      classified[version]['epochs'] = current['epochs']
      classified[version]['loss'] = current['loss']
      classified[version]['delta_loss'] = current['delta_loss']
      classified[version]['delta_acc'] = current['delta_acc']
  

endPath = '/Users/caine2003/Documents/CS591LFDTermProject/classified_runs/'

for i in classified:
  print("version: %s" % i)
  absPath = endPath + ('version_%s_size_%d_run_%s.csv' % (i,(int(i)*3),classified[i]['run']))
  for key in classified[i]:
    if key == 'epochs':
      with open(absPath, 'w') as csvFile:
        info = csv.writer(csvFile,delimiter=',')
        info.writerows(classified[i][key])
    elif key == 'run':
      print("%s:\t%s" % (key, classified[i][key]))
    else:
      print("%s:\t%f" % (key, float(classified[i][key])))





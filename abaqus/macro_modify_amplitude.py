# -*- coding: mbcs -*-
# Do not delete the following import lines
import __main__
import csv
from abaqus import *
from abaqusConstants import *

def read_csv(file, header=0):
	with open(file, 'r') as f:
		reader = csv.reader(f)

		content = []
		for i, row in enumerate(reader):
			if i<=header:
				continue
			content.append(row)
	return content

def mod_amplitude(model, amp, data):
	mdb.models[model].amplitudes[amp].setValues(timeSpan=STEP, smooth=SOLVER_DEFAULT, data=data)

def mod_BC(model, amp, data, on='u1'):
	u1 = u2 = u3 = UNSET
	if on=='u1':
		u1 = data
	if on=='u2':
		u2 = data
	if on=='u3':
		u3 = data
	mdb.models[model].boundaryConditions[amp].setValues(u1=u1, u2=u2, u3=u3, amplitude=amp)


# >------------------------------------------------------------
# modify amplitude
model = 'SEshearxy'
file = model+'/morient_20_750_shearxy_load.csv'

print('<<< model: '+model+' >>>')
print('<<< load: '+file+' >>>')
print('<<< modify amplitude >>>')
amps = ['Ax', 'Ay', 'Az', 
		'Bx', 'By', 'Bz', 
		'Cx', 'Cy', 'Cz', 
		'Dx', 'Dy', 'Dz', 
		'Ex', 'Ey', 'Ez', 
		'Fx', 'Fy', 'Fz', 
		'Gx', 'Gy', 'Gz', 
		'Hx', 'Hy', 'Hz']
content = read_csv(file, header=1)
for i, amp in enumerate(amps):
	data = [[float(row[0]), float(row[i+1])] for row in content]
	mod_amplitude(model, amp, data)

# >------------------------------------------------------------
# modify BCs
print('<<< modify BCs >>>')
max_val = read_csv(file, header=0)[0]
for i, amp in enumerate(amps):
	on = ['u1', 'u2', 'u3']
	data = float(max_val[i+1])
	mod_BC(model, amp, data, on[i%3])


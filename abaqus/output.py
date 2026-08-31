# >----------------------------------------------------------------------------------------------------
# > abaqus python output macro
# > @documentation: https://docs.abqpy.com/en/latest/index.html
# >----------------------------------------------------------------------------------------------------
import os
from abaqus import *
from abaqusConstants import *
from viewerModules import *

job_name = 'DBneural'
part_name = 'DogBone-1'.upper()
output_root = os.getcwd()
directions = ['11', '22', '33', '12', '13', '23']
ele_num = 1

ODB_NAME = job_name+'.odb'
if not os.path.exists(job_name+'.odb'):
	raise ValueError(ODB_NAME+' not exists !')

if not os.path.exists(output_root+'/result'):
	os.mkdir(output_root+'/result')

o2 = session.openOdb(name=ODB_NAME)
session.viewports['Viewport: 1'].setValues(displayedObject=o2)
odb = session.odbs[ODB_NAME]

# >------------------------------------------------------------
# element stress-strain
for direction in directions:
	session.xyDataListFromField(odb=odb, outputPosition=ELEMENT_CENTROID, 
		variable=(
			('LE', INTEGRATION_POINT, ((COMPONENT, 'LE'+direction), )), 
			('S', INTEGRATION_POINT,((COMPONENT, 'S'+direction), )), 
		), elementSets=('ALL ELEMENTS', ), )

	strain = []; stress = []
	for i in range(ele_num):
		strain.append(session.xyDataObjects['LE:LE'+direction+' PI: '+part_name+' E: '+str(i+1)+' Centroid'])
		stress.append(session.xyDataObjects['S:S'+direction+' PI: '+part_name+' E: '+str(i+1)+' Centroid'])

	# >---------------------------------------------
	# avg
	LE_avg = avg(tuple(strain))
	LE_avg.setValues(sourceDescription='LE avg')
	LE_key = 'LE'+direction
	session.xyDataObjects.changeKey(LE_avg.name, LE_key)
	LE_obj = session.xyDataObjects[LE_key]

	S_avg = avg(tuple(stress))
	S_avg.setValues(sourceDescription='S avg')
	S_key = 'S'+direction
	session.xyDataObjects.changeKey(S_avg.name, S_key)
	S_obj = session.xyDataObjects[S_key]
	# >---------------------------------------------
	# combine
	LE_S = combine(LE_obj, S_obj)
	LE_S.setValues(sourceDescription='combine ( "LE","S" )')
	LE_S_key = 'LE-S'+direction
	session.xyDataObjects.changeKey(LE_S.name, LE_S_key)
	LE_S_obj = session.xyDataObjects[LE_S_key]

	output_name = job_name+'_LE-S'+direction+'.txt'
	output_path = output_root+'/result/'+output_name
	session.writeXYReport(fileName=output_path, xyData=(LE_S_obj, ),appendMode=OFF)

# >------------------------------------------------------------
# nodal displacement
directions = ['1', '2', '3']
for direction in directions:
	session.xyDataListFromField(odb=odb, outputPosition=NODAL, 
		variable=(
			('U', NODAL, ((COMPONENT, 'U'+direction), )), 
		), nodeLabels=((part_name, 3369), ), )

	U = []
	for i in range(ele_num):
		# with open('./debug.log', 'w') as f:
		# 	f.write(str(session.xyDataObjects.keys()))
		U.append(session.xyDataObjects['U:U'+direction+' PI: '+part_name+' N: '+str(3369)])

	# >---------------------------------------------
	# avg
	U_avg = avg(tuple(U))
	U_avg.setValues(sourceDescription='U avg')
	U_key = 'U'+direction
	session.xyDataObjects.changeKey(U_avg.name, U_key)
	U_obj = session.xyDataObjects[U_key]

	output_name = job_name+'_U'+direction+'.txt'
	output_path = output_root+'/result/'+output_name
	session.writeXYReport(fileName=output_path, xyData=(U_obj, ),appendMode=OFF)

# >---------------------------------------------
# viewport angle
# session.viewports['Viewport: 1'].view.setValues(session.views['Left'])
# session.viewports['Viewport: 1'].view.setValues(session.views['Iso'])
session.viewports['Viewport: 1'].view.setValues(session.views['Front'])
session.viewports['Viewport: 1'].view.rotate(zAngle=90)

# >---------------------------------------------
# viewport annotation
session.viewports['Viewport: 1'].viewportAnnotationOptions.setValues(title=OFF, state=OFF, annotations=OFF, compass=OFF)
session.viewports['Viewport: 1'].viewportAnnotationOptions.setValues(legendPosition=(2, 98), legendBox=False, legendDecimalPlaces=2)
session.viewports['Viewport: 1'].odbDisplay.commonOptions.setValues(deformationScaling=UNIFORM, uniformScaleFactor=5)
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(CONTOURS_ON_DEF, ))
session.viewports['Viewport: 1'].odbDisplay.contourOptions.setValues(contourStyle=CONTINUOUS)
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(variableLabel='S', outputPosition=INTEGRATION_POINT, refinement=(COMPONENT, 'S22'), )

# >---------------------------------------------
# screenshot
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=0)
session.printOptions.setValues(vpDecorations=OFF)
session.pngOptions.setValues(imageSize=(4000,4000))
session.printToFile(
        fileName=output_root+'/result/'+job_name, format=PNG,
        canvasObjects=(session.viewports['Viewport: 1'], )
)

# >---------------------------------------------
# animation
session.animationController.setValues(animationType=TIME_HISTORY, viewports=('Viewport: 1', ))
session.animationController.play(duration=UNLIMITED)
session.imageAnimationOptions.setValues(vpDecorations=OFF, vpBackground=OFF, compass=OFF)
session.writeImageAnimation(
	fileName=output_root+'/result/'+job_name, 
	format=AVI, canvasObjects=(session.viewports['Viewport: 1'], )
)





# >----------------------------------------------------------------------------------------------------
# > LEGACY 
# >----------------------------------------------------------------------------------------------------
# job_name = 'AA6111_7min_298K_abq-test'
# part_name = 'PART-1-1'
# output_path = os.getcwd()
# direction = '11'
# ele_num=20*20*20


# if os.path.exists(job_name+'.odb'):
# 	ODB_NAME = job_name+'.odb'
# else:
# 	ODB_NAME = './cache/results/abaqus/'+job_name+'/'+job_name+'.odb'

# o2 = session.openOdb(name=ODB_NAME)
# session.viewports['Viewport: 1'].setValues(displayedObject=o2)

# odb = session.odbs[ODB_NAME]

# session.xyDataListFromField(odb=odb, outputPosition=ELEMENT_CENTROID, 
# 	variable=(
# 	('LE', INTEGRATION_POINT, ((COMPONENT, 'LE'+direction), )), 
# 	('S', INTEGRATION_POINT,((COMPONENT, 'S'+direction), )),
# 	# ('SDV124', INTEGRATION_POINT),('SDV125', INTEGRATION_POINT), 									# rhos, rhog
# 	# ('SDV1', INTEGRATION_POINT), ('SDV2', INTEGRATION_POINT), ('SDV3', INTEGRATION_POINT),        # slip resistance 
# 	# ('SDV4', INTEGRATION_POINT), ('SDV5', INTEGRATION_POINT), ('SDV6', INTEGRATION_POINT), 
# 	# ('SDV7', INTEGRATION_POINT), ('SDV8', INTEGRATION_POINT), ('SDV9', INTEGRATION_POINT), 
# 	# ('SDV10', INTEGRATION_POINT), ('SDV11', INTEGRATION_POINT), ('SDV12', INTEGRATION_POINT),
# 	# ('SDV25', INTEGRATION_POINT), ('SDV26', INTEGRATION_POINT), 
# 	# ('SDV27', INTEGRATION_POINT), ('SDV28', INTEGRATION_POINT),
# 	), elementSets=(' ALL ELEMENTS', ), )

# strain = []
# stress = []
# # rhos = []
# # rhog = []
# # sdv25 = []; sdv26 = []; sdv27 = []; sdv28 = [];
# for i in range(ele_num):
# 	strain.append(session.xyDataObjects['LE:LE'+direction+' PI: '+part_name+' E: '+str(i+1)+' Centroid'])
# 	stress.append(session.xyDataObjects['S:S'+direction+' PI: '+part_name+' E: '+str(i+1)+' Centroid'])
# 	# rhos.append(session.xyDataObjects['SDV124 PI: '+part_name+' E: '+str(i+1)+' Centroid'])
# 	# rhog.append(session.xyDataObjects['SDV125 PI: '+part_name+' E: '+str(i+1)+' Centroid'])
# 	# sdv25.append(session.xyDataObjects['SDV25 PI: '+part_name+' E: '+str(i+1)+' Centroid'])
# 	# sdv26.append(session.xyDataObjects['SDV26 PI: '+part_name+' E: '+str(i+1)+' Centroid'])
# 	# sdv27.append(session.xyDataObjects['SDV27 PI: '+part_name+' E: '+str(i+1)+' Centroid'])
# 	# sdv28.append(session.xyDataObjects['SDV28 PI: '+part_name+' E: '+str(i+1)+' Centroid'])

# LE_avg = avg(tuple(strain))
# LE_avg.setValues(sourceDescription='LE avg')
# session.xyDataObjects.changeKey(LE_avg.name, 'LE')

# S_avg = avg(tuple(stress))
# S_avg.setValues(sourceDescription='S avg')
# session.xyDataObjects.changeKey(S_avg.name, 'S')

# # sdv25_avg = avg(tuple(sdv25))
# # sdv25_avg.setValues(sourceDescription='sdv25 avg')
# # session.xyDataObjects.changeKey(sdv25_avg.name, 'sdv25')

# # sdv26_avg = avg(tuple(sdv26))
# # sdv26_avg.setValues(sourceDescription='sdv26 avg')
# # session.xyDataObjects.changeKey(sdv26_avg.name, 'sdv26')

# # sdv27_avg = avg(tuple(sdv27))
# # sdv27_avg.setValues(sourceDescription='sdv27 avg')
# # session.xyDataObjects.changeKey(sdv27_avg.name, 'sdv27')

# # sdv28_avg = avg(tuple(sdv28))
# # sdv28_avg.setValues(sourceDescription='sdv28 avg')
# # session.xyDataObjects.changeKey(sdv28_avg.name, 'sdv28')

# xy1 = session.xyDataObjects['LE']
# xy2 = session.xyDataObjects['S']
# xy3 = combine(xy1, xy2)
# xy3.setValues(sourceDescription='combine ( "LE","S" )')
# session.xyDataObjects.changeKey(xy3.name, 'LE-S')
# x0 = session.xyDataObjects['LE-S']
# outputss = 'StressStrain_'+job_name+'.txt'	
# outputss = output_path+'/'+outputss
# session.writeXYReport(fileName=outputss, xyData=(x0, ),appendMode=OFF)

# # xy4 = session.xyDataObjects['Rhos']
# # xy5 = session.xyDataObjects['Rhog']
# # xy6 = combine(xy4, xy5)
# # outputrho = 'Rho_'+job_name+'.txt'	
# # outputrho = output_path+'/'+outputrho
# # session.writeXYReport(fileName=outputrho, xyData=(xy6, ),appendMode=OFF)

# # xy7 = session.xyDataObjects['sdv25']
# # xy8 = session.xyDataObjects['sdv26']
# # xy9 = combine(xy7, xy8)
# # outputrss = 'rss_1_2_'+job_name+'.txt'	
# # outputrss = output_path+'/'+outputrss
# # session.writeXYReport(fileName=outputrss, xyData=(xy9, ),appendMode=OFF)

# # xy10 = session.xyDataObjects['sdv27']
# # xy11 = session.xyDataObjects['sdv28']
# # xy12 = combine(xy10, xy11)
# # outputrss = 'rss_3_4_'+job_name+'.txt'	
# # outputrss = output_path+'/'+outputrss
# # session.writeXYReport(fileName=outputrss, xyData=(xy12, ),appendMode=OFF)

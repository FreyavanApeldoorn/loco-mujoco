import opensim as osim
import mujoco

# load model
# missing vtp files:
# pelvis, fibula_r, talus_rv, foot, bofoot, fibula_l, talus_lv
model_path='working_folder/models/Body_model_opensim.osim'
osim_model = osim.Model(model_path)

# Get actuators
actuators = osim_model.getActuators()

# Create controller
brain = osim.PrescribedController()

#I kinda want to avoid loops?? There must be a better way for this
for i in range(actuators.getSize()):
    brain.addActuator(actuators.get(i))

osim_model.addController(brain)

print([m.getName() for m in osim_model.getActuators()])

print([m.getName() for m in osim_model.getActuators()] == [m.getName() for m in osim_model.getMuscles()])

# state = osim_model.initSystem()
'''
brain = osim.PrescribedController()
brain.addActuator(biceps)
brain.prescribeControlForActuator("biceps",
                                  osim.StepFunction(0.5, 3.0, 0.3, 1.0))
'''

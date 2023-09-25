COOK = ['grab','heat','plate','pour']
CLEAN = ['clean', 'wipe']
MAS = ['go_to', 'give_massage']
def affordance_score2(action, task = 'cook'):
  if 'done' in action: return 2
  action = action.replace('robot action: robot.','').split("(")[0].lower()
  print(action)
  if task == 1:
    if action not in COOK: return 0
    else: return 1
  elif task == 2:
    if action not in CLEAN: return 0
    else: return 1
  else:
    if action not in MAS: return 0
    else: return 1

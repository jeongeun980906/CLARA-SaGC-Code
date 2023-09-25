import copy
PROMPT_STARTER = {
    'cook': "I am a cooking robot. Your possible action is grab, heat and plate",
    'clean': "I am a cleaning robot. Your possible action is clean and wipe",
    'mas': "I am a massage robot, Your possible action is go_to and give_massage"
}

PROMPTS_COOK = [
        """"
        objects: [bread, pan, table, water] 
        goal: cook me toast 
        robot action: robot.grab(bread) 
        robot action: robot.heat(bread) 
        robot action: robot.plate(bread)  
        robot action: robot.done()
        """,
        """
        objects: [table, watter, coffemachine] 
        goal: give a person wearing black shirt a coffee 
        robot action: robot.grab(coffee) 
        robot action: robot.pour(coffee)  
        robot action: robot.done()
        """,
        """
        objects: [bread, pan, bacon, watter, table] 
        goal: give me a sandwich 
        robot action: robot.grab(bread) 
        robot action: robot.heat(bread) 
        robot action: robot.grab(bacon) 
        robot action: robot.heat(bacon) 
        robot action: robot.plate(bread) 
        robot action: robot.plate(bacon) 
        robot action: robot.done()
        """,
        """
        objects: [water, table, pan, bread]
        goal: I am thirsty
        robot action: robot.grab(water)
        robot action: robot.pour(water)
        robot action: robot.done()
        """
        ]


PROMPTS_CLEAN = [
        """"
        scene: [kitchen, living room, bedroom, bathroom] 
        objects: [table, pan, bread] 
        goal: clean the kitchen
        robot action: robot.clean(kitchen)
        robot action: robot.done()
        """,
        """
        scene: [kitchen, living room, bedroom]
        objects: [table, pan, bread] 
        goal: clean bedroom
        robot action: robot.clean(bedroom)
        robot action: robot.done()
        """,
        """
        scene: [kitchen, living room, bedroom] 
        objects: [table, pan, bread] 
        goal: clean table
        robot action: robot.wipe(table)
        robot action: robot.done()
        """,
        """
        scene: [kitchen, living room, bedroom] 
        objects: [table, pan, desk, bed] 
        goal: clean desk
        robot action: robot.wipe(desk)
        robot action: robot.done()
        """
        ]

PROMPTS_MAS = [
        """"
        people: [person wearing black shirt, person wearing yellow shirt, person wearing red shirt] 
        goal: Give a massage to a person wearing black shirt
        robot action: robot.go_to(person wearing black shirt)
        robot action: robot.give_massage(person wearing black shirt)
        robot action: robot.done()
        """,
        """
        poeple: [person wearing green shirt, person wearing green shirt, person wearing blue shirt]
        goal: Give a massage to a person wearing green shirt
        robot action: robot.go_to(person wearing green shirt)
        robot action: robot.give_massage(person wearing green shirt)
        robot action: robot.done()
        """,
        """
        poeple: [person wearing green shirt, person wearing green shirt, person wearing blue shirt]
        goal: A person wearing green shirt needs a massage
        robot action: robot.go_to(person wearing green shirt)
        robot action: robot.give_massage(person wearing green shirt)
        robot action: robot.done()
        """,
        """
        people: [person wearing black shirt, person wearing yellow shirt, person wearing red shirt] 
        goal: A person wearing yellow shirt needs a massage
        robot action: robot.go_to(person wearing yellow shirt)
        robot action: robot.give_massage(person wearing yellow shirt)
        robot action: robot.done()
        """
        ]


def get_prompts(example=False, task = 'cook'):
    if task == 'cook':
        prompts = copy.deepcopy(PROMPTS_COOK)
        if example:
            prompts[-1] = """
            objects: [water, table, pan, bread, 'coffee]
        goal: I am thirsty
        robot thought: I am not sure what to do grab first
        question: What should I do first?
        answer: grab water
        robot action: robot.grab(water)
        robot action: robot.pour(water)
        robot action: robot.done()"""
            prompts.append("""
            objects: [water, table, pan, bread, coffee]
        goal: wipe here
        robot thought: I can not do this task
        question: provide more information about what to do
        answer: I can not do this task
        robot action: robot.done()
            
            """)
    elif task == 'clean':
        prompts = copy.deepcopy(PROMPTS_CLEAN)
        if example:
            prompts[-1] = """
        scene: [kitchen, living room, bedroom] 
        objects: [table, pan, desk, bed] 
        goal: clean here
        robot thought: this is uncertain because I am not sure what is here
        question: what is here
        answer: desk
        robot action: robot.wipe(desk)
        robot action: robot.done()"""
            prompts.append("""
            objects: [water, table, pan, bread, coffee]
        goal: heat bread
        robot thought: I can not do this task
        question: provide more information about what to do
        answer: I can not do this task
        robot action: robot.done()
            """)
    elif task == 'mas':
        prompts = copy.deepcopy(PROMPTS_MAS)
        if example:
            prompts[-1] = """
        people: [person wearing black shirt, person wearing yellow shirt, person wearing red shirt] 
        goal: Someone needs a massage
        robot thought: this is uncertain because I am not sure who needs a massage
        question: who needs a massage
        answer: person wearing yellow shirt
        robot action: robot.go_to(person wearing yellow shirt)
        robot action: robot.give_massage(person wearing yellow shirt)
        robot action: robot.done()"""
            prompts.append("""
        people: [person wearing black shirt, person wearing yellow shirt, person wearing red shirt] 
        goal: heat bread
        robot thought: I can not do this task
        question: provide more information about what to do
        answer: I can not do this task
        robot action: robot.done()
            """)
    return prompts

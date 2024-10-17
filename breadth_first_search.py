import matplotlib.pyplot as plt
def BFS():
    visited_queue=[]
    not_visited_queue=[]
    print(len(not_visited_queue))
    while True:
        queue=input("Please Enter your graph or tree vertex : ")
        if queue=='exit':
            break
        else:
            not_visited_queue.append(queue)
    print(not_visited_queue)
    l=(len(not_visited_queue))
    while True:
        if len(not_visited_queue)==0:
            break
        visited_queue.append(not_visited_queue[0])
    
        print("visited ",visited_queue)
        not_visited_queue.pop(0)
        print("Not visited",not_visited_queue)


        plt.figure(figsize=(12, 8))
        plt.plot(visited_queue, visited_queue, marker='o', label='Visited Nodes', color='blue')
        plt.title('BFS Visited Nodes')
        plt.xlabel('Step')
        plt.ylabel('Nodes')
        plt.xticks(visited_queue, visited_queue, rotation=45)
        plt.grid()
        plt.legend()
            
        
        plt.show()  # Show the plot after BFS completes



BFS()


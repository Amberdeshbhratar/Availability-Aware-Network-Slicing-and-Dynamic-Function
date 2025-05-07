import itertools
import gymnasium as gym
import json
from gymnasium import error, spaces, utils
import numpy as np
import random
import math



# Global vars
links = []
capacity = {}  # link capacity in Gbps
delay = {}  # link delay in ms
crs = [] # CR CPU capacity # of cores
paths = {}
all_paths = {}
rus = []

tow = [3,2,2]

class DRC:
    def __init__(self, id, cpu_CU, cpu_DU, cpu_RU, ram_CU, ram_DU, ram_RU, Fs_CU, Fs_DU, Fs_RU, delay_BH, delay_MH,
                 delay_FH, bw_BH, bw_MH, bw_FH):
        self.id = id

        self.cpu_CU = cpu_CU
        self.ram_CU = ram_CU
        self.Fs_CU = Fs_CU

        self.cpu_DU = cpu_DU
        self.ram_DU = ram_DU
        self.Fs_DU = Fs_DU

        self.cpu_RU = cpu_RU
        self.ram_RU = ram_RU
        self.Fs_RU = Fs_RU

        self.latency_BH = delay_BH
        self.latency_MH = delay_MH
        self.latency_FH = delay_FH

        self.bw_BH = bw_BH
        self.bw_MH = bw_MH
        self.bw_FH = bw_FH


class CR:
    def __init__(self, id, cpu, num_BS, rus:None):
        self.id = id
        self.cpu = cpu
        self.num_BS = num_BS
        self.rus = rus

    def __str__(self):
        return "ID: {}\tCPU: {}".format(self.id, self.cpu)

def DRC_structure_T1():
    DRC1 = DRC(1, 0, 0, 4.9, 0, 0, 0.01, [0], [0], ['f8', 'f7', 'f6', 'f5', 'f4', 'f3', 'f2', 'f1', 'f0'], 0, 0, 10, 0, 0, 3)
    DRC2 = DRC(2, 0, 0.49, 4.41, 0, 0.01, 0.01, [0], ['f8'], ['f7', 'f6', 'f5', 'f4', 'f3', 'f2', 'f1', 'f0'], 0, 10, 10, 0, 3, 5.4)
    DRC3 = DRC(3, 0, 3, 3.92, 0, 0.01, 0.01, [0], ['f8', 'f7'], ['f6', 'f5', 'f4', 'f3', 'f2', 'f1', 'f0'], 0, 10, 10, 0, 3, 5.4)
    DRC4 = DRC(4, 0, 1.71, 3.185, 0, 0.01, 0.01, [0], ['f8', 'f7', 'f6', 'f5', 'f4', 'f3'], ['f2', 'f1', 'f0'], 0, 10, 0.25, 0, 3, 5.6)
    DRC5 = DRC(5, 0, 2.54, 2.354, 0, 0.01, 0.01, [0], ['f8', 'f7', 'f6', 'f5', 'f4', 'f3', 'f2'], ['f1', 'f0'], 0, 10, 0.25, 0, 3, 17.4)
    DRC6 = DRC(6, 0.49, 1.225, 3.185, 0.01, 0.01, 0.01, ['f8'], ['f7', 'f6', 'f5', 'f4', 'f3'], ['f2', 'f1', 'f0'], 10, 10, 0.25, 3, 5.4, 5.6)
    DRC7 = DRC(7, 0.98, 0.735, 3.185, 0.01, 0.01, 0.01, ['f8', 'f7'], ['f6', 'f5', 'f4', 'f3'], ['f2', 'f1', 'f0'], 10, 10, 0.25, 3, 5.4, 5.6)
    DRC8 = DRC(8, 0.49, 2.058, 2.352, 0.01, 0.01, 0.01, ['f8'], ['f7', 'f6', 'f5', 'f4', 'f3', 'f2'], ['f1', 'f0'], 10, 10, 0.25, 3, 5.4, 17.4)
    DRC9 = DRC(9, 0.98, 1.568, 2.352, 0.01, 0.01, 0.01, ['f8', 'f7'], ['f6', 'f5', 'f4', 'f3', 'f2'], ['f1', 'f0'], 10, 10, 0.25, 3, 5.4, 17.4)   
    DRCs = {0: DRC1, 1: DRC2, 2: DRC3, 3: DRC4, 4: DRC5, 5: DRC6, 6: DRC7, 7: DRC8, 8: DRC9}
    return DRCs
    
class Graph:
    def __init__(self):
        # Dictionary to store nodes with their properties: {node: {"capacity": value, "flag": value}}
        self.nodes = {}
        # Dictionary to store edges with their properties: {(node1, node2): {"bandwidth": value, "latency": value}}
        self.edges = {}

    def add_node(self, node, capacity, flag):
        # Add a node with a given capacity and flag
        if node not in self.nodes:
            self.nodes[node] = {"capacity": capacity, "flag": flag}
        else:
            print(f"Node {node} already exists.")

    def add_edge(self, node1, node2, bandwidth, latency, values):
        # Ensure both nodes exist before adding an edge
        if node1 not in self.nodes or node2 not in self.nodes:
            print(f"One or both nodes {node1}, {node2} do not exist.")
            return
        
        # Add an edge between node1 and node2 with bandwidth, latency, and values array
        self.edges[(node1, node2)] = {"bandwidth": bandwidth, "latency": latency, "values": values}
        # self.edges[(node2, node1)] = {"bandwidth": bandwidth, "latency": latency, "values": values}  # Undirected graph
    
    def add_edge_value(self,node1,node2,value):
        # Add a value to the 'values' array of the edge (node1, node2)
        # if (node1, node2) in self.edges:
        # print("Adding ")
        # print(value)
        # print(self.edges[(node1,node2)])
        if value not in self.edges[(node1,node2)]["values"]:
            self.edges[(node1,node2)]["values"].append(value)
        # if value not in self.edges[(node2,node1)]["values"]:
        #     self.edges[(node2,node1)]["values"].append(value) 
        # print(self.edges[(node1,node2)]["values"]) # For undirected graph, add in both directions
        # else:
        #     print(f"Edge between {node1} and {node2} does not exist.")

    def get_edge(self,node1,node2):
        return self.edges[(node1,node2)]
    
    def get_node(self,node):
        return self.nodes[node]
    
    def update_capacity(self,node,val):
        self.nodes[node]["capacity"] -= val
    
    def update_bw_latency(self,node1,node2,bw,latency):
        self.edges[(node1,node2)]["bandwidth"] -= bw
        self.edges[(node1,node2)]["latency"] -= latency

        # self.edges[(node2,node1)]["bandwidth"] -= bw
        # self.edges[(node2,node1)]["latency"] -= latency

    def get_neighbors(self, node):
        # Return neighbors of a given node based on edges
        neighbors = [n2 for (n1, n2) in self.edges.keys() if n1 == node]
        return neighbors
    
    def get_paths(self,curr,source,target,vis,p):
        vis[curr] = 1
        if curr == target:
            p.pop()
            return 1
        else:
            neighbours = self.get_neighbors(curr)
            for i in neighbours:
                if vis[i] == 0:
                    vis[i]=1
                    p.append(i)
                    if self.get_paths(i,source,target,vis,p):
                        return 1
                    vis[i] = 0
                    p.pop()
        return 0
    
    def dfs(self,curr,tar,curr_path,total_paths):
        if curr == tar:
            total_paths.append(list(curr_path))
            return
        curr_path.append(curr)
        neighbours = self.get_neighbors(curr)
        for i in neighbours:
            self.dfs(i,tar,curr_path,total_paths)
            if i != tar:
             curr_path.pop()


    def display(self):
        # Print the nodes with their properties
        print("Nodes:")
        for node, props in self.nodes.items():
            print(f"{node}: Capacity={props['capacity']}, Flag={props['flag']}")
        
        # Print the edges with their properties
        print("\nEdges:")
        for (node1, node2), props in self.edges.items():
            print(f"{node1} -- {node2}: Bandwidth={props['bandwidth']}, Latency={props['latency']}")


class Sliavalilran(gym.Env):

    def __init__(self):
        
        self.links_file = '16_CRs_links.json'
        self.nodes_file = '16_CRs_nodes.json'

        self.wavelengths = []
        self.nodes_activated = []
        
        self.read_topology()

        self.graph = Graph()

        tn = 0

       
        for i in crs:
            self.graph.add_node(i[0],i[1],i[2])
            tn = max(tn,i[0])
            if i[2] == 1:
                rus.append(i[0])
        
    
       
        for i in links:
            # print(i[0]," ",i[1])
            self.graph.add_edge(node1=i[0],node2=i[1],bandwidth=i[2],latency=i[3],values=[])
            # print(i)
        
        self.done = False
        self.end_ep = False
        # print(tn+1)

        for i in crs:
            if i[2]==1:
                curr_path = []
                total_paths = []
                self.graph.dfs(i[0],0,curr_path,total_paths)
                all_paths[i[0]] = total_paths
        # print(all_paths)


        # for i in crs:
        #     if i[2] == 1:
        #         tow_path = []
        #         vis = [0]*(tn+1)
        #         p = [1,2,3]
        #         while len(p) == 3:
        #             p = []
        #             vis[i[0]]=0
        #             vis[0]=0
        #             p.append(i[0])
        #             self.graph.get_paths(i[0],i[0],0,vis,p)
        #             # print(p)
        #             p.reverse()
        #             if len(p) == 3:
        #                 tow_path.append(p)
        #         all_paths[i[0]] = tow_path
        # print(all_paths)

        self.num_nodes = len(crs)
        self.num_links = len(links)
        self.num_vncs = 9
        self.num_wavelengths = 11
        self.num_rus = len(rus)
        # print(self.num_nodes)
        # Example: Observation is a flattened vector of node and link features
        low = np.concatenate([np.zeros(self.num_nodes), np.zeros(self.num_links),np.zeros(self.num_links),np.zeros(self.num_links*self.num_wavelengths)])  # Lower bound: 0 for both ranges
        high = np.concatenate([np.full(self.num_nodes, 64), np.full(self.num_links, 1000),np.full(self.num_links, 0.2),np.full(self.num_links*self.num_wavelengths, 11)])  # Upper bound: [0,1000] for first r, [0,32] for next n

        self.observation_space = spaces.Box(
            low=low, high=high, dtype=np.float32
        )
        
        # Action space: VNC (0 to num_vncs-1), wavelength (0 to num_wavelengths-1), path (0 to num_paths-1)
        actions = []
        for i in crs:
            if i[2] == 1:
                actions.append(self.num_vncs)
                actions.append(len(all_paths[i[0]]))
                actions.append(len(all_paths[i[0]]))
                actions.append(len(all_paths[i[0]]))
                actions.append(self.num_wavelengths)
                actions.append(3)
        # self.action_space = spaces.MultiDiscrete([self.num_vncs,  self.num_rus,self.num_wavelengths,3])
        self.action_space = spaces.MultiDiscrete(actions)

        self.state,info = self.reset()
        

        # print("info")
        # print(info)
        # print(self.state)

        # print(self.graph.nodes)
        # print(self.graph.edges)

    def reset(self,seed=None, options=None):
        self.graph = Graph()

        for i in crs:
            self.graph.add_node(i[0],i[1],i[2])
        
        for i in links:
            self.graph.add_edge(node1=i[0],node2=i[1],bandwidth=i[2],latency=i[3],values=[])
            # print(i)
        self.wavelengths = []
        self.nodes_activated = []
        self.done = False
        self.end_ep = False

        self.node_cpu = []
        
        for i in crs:
            self.node_cpu.append(i[1])
        self.link_bandwidth = []  
        self.link_latency = []
        self.link_wavelength = []

        for i in links:
            self.link_bandwidth.append(i[2])
            self.link_latency.append(i[3])
            self.link_wavelength.append([0]*self.num_wavelengths)
        
        self.state = np.concatenate(
            [self.node_cpu, self.link_bandwidth, self.link_latency, np.array(self.link_wavelength).flatten()], dtype=np.float32
        )
        # print(self.state)
        return self.state,{}

        # return self.graph

    
    def constraints(self,path,vnc,wavelength):

        DRCs = DRC_structure_T1()
        VNC = DRCs[vnc]
        # p1 -> RU p2 -> DU p3 -> CU
        p1 = path[0]
        p2 = path[1]
        p3 = path[2]



        if (VNC.cpu_RU > self.graph.get_node(p1)["capacity"]) or  (VNC.cpu_DU > self.graph.get_node(p2)["capacity"]) or  (VNC.cpu_CU > self.graph.get_node(p3)["capacity"]) :
            return False
        
        if (VNC.bw_BH > self.graph.get_edge(p3,0)["bandwidth"]) or (VNC.bw_MH > self.graph.get_edge(p2,p3)["bandwidth"]) or (VNC.bw_FH > self.graph.get_edge(p1,p2)["bandwidth"]):
            return False

        if (VNC.latency_BH < self.graph.get_edge(p3,0)["latency"]) or (VNC.latency_MH < self.graph.get_edge(p2,p3)["latency"]) or (VNC.latency_FH < self.graph.get_edge(p1,p2)["latency"]):
            return False
        # print("wavelength vector ",self.graph.get_edge(p1,0)['values'])
        if (wavelength in self.graph.get_edge(p3,0)['values']) or (wavelength in self.graph.get_edge(p2,p3)['values']) or (wavelength in self.graph.get_edge(p1,p2)['values']) :
            return False
        
        return True
    
        
    
    def is_disjoint(self,path1,path2):
        if path1[1] == path2[1] or path1[2] == path2[2]:
            return False
        return True


    
    def validate_action(self,action,cnt):
        # VNC,path,wavelength = action
        # vnc = action["VNC"]
        # DRCs = DRC_structure_T1()
        # VNC = DRCs[vnc]
        # path = action["path"]
        # wavelength = action["wavelength"]
        # DRC = DRC_structure_T1()


        vnc,primary_path_idx,backup1_path_idx,backup2_path_idx,wavelength,request_type = action

        primary_path = all_paths[rus[cnt]][primary_path_idx]
        backup1_path = all_paths[rus[cnt]][backup1_path_idx]
        backup2_path = all_paths[rus[cnt]][backup2_path_idx]


        DRCs = DRC_structure_T1()
        VNC = DRCs[vnc]


        # # p1 -> RU p2 -> DU p3 -> CU
        # p1 = path[0]
        # p2 = path[1]
        # p3 = path[2]



        # if (VNC.cpu_RU > self.graph.get_node(p1)["capacity"]) or  (VNC.cpu_DU > self.graph.get_node(p2)["capacity"]) or  (VNC.cpu_CU > self.graph.get_node(p3)["capacity"]) :
        #     return False
        
        # if (VNC.bw_BH > self.graph.get_edge(p3,0)["bandwidth"]) or (VNC.bw_MH > self.graph.get_edge(p2,p3)["bandwidth"]) or (VNC.bw_FH > self.graph.get_edge(p1,p2)["bandwidth"]):
        #     return False

        # # if (VNC.latency_BH > self.graph.get_edge(0,p1)["latency"]) or (VNC.latency_MH > self.graph.get_edge(p1,p2)["latency"]) or (VNC.latency_FH > self.graph.get_edge(p2,p3)["latency"]):
        # #     return False
        # # print("wavelength vector ",self.graph.get_edge(p1,0)['values'])
        # if (wavelength in self.graph.get_edge(p3,0)['values']) or (wavelength in self.graph.get_edge(p2,p3)['values']) or (wavelength in self.graph.get_edge(p1,p2)['values']) :
        #     return False

        if request_type == 0:
            if not(self.is_disjoint(primary_path,backup1_path) and self.is_disjoint(primary_path,backup2_path) and self.is_disjoint(backup1_path,backup2_path)):
                return False
            if self.constraints(primary_path,vnc,wavelength) and self.constraints(backup1_path,vnc,wavelength) and self.constraints(backup2_path,vnc,wavelength) == False:
                return False
        else:
            if not(self.is_disjoint(primary_path,backup1_path)):
                return False
            if self.constraints(primary_path,vnc,wavelength) and self.constraints(backup1_path,vnc,wavelength) == False:
                return False


        return True
    
    def update_state(self,path,vnc,wavelength):
        DRCs = DRC_structure_T1()
        VNC = DRCs[vnc]

        # p1->RU p2->DU p3->CU
        p1 = path[0]
        p2 = path[1]
        p3 = path[2]

        self.graph.update_capacity(p1,VNC.cpu_RU)
        self.graph.update_capacity(p2,VNC.cpu_DU)
        self.graph.update_capacity(p3,VNC.cpu_CU)

        self.graph.update_bw_latency(p3,0,VNC.bw_BH,VNC.latency_BH)
        self.graph.update_bw_latency(p2,p3,VNC.bw_MH,VNC.latency_MH)
        self.graph.update_bw_latency(p1,p2,VNC.bw_FH,VNC.latency_FH)


        # print(p1,p2,p3)

        self.graph.add_edge_value(p3,0,wavelength)
        self.graph.add_edge_value(p2,p3,wavelength)
        self.graph.add_edge_value(p1,p2,wavelength)

        node_activated = 0

        if p1 not in self.nodes_activated:
            node_activated+=0.2
            self.nodes_activated.append(p1)
        if p2 not in self.nodes_activated:
            node_activated+=0.2
            self.nodes_activated.append(p2)
        if p3 not in self.nodes_activated:
            node_activated+=0.2
            self.nodes_activated.append(p3)
        
        return node_activated
    

           
    def step(self,action):

        actions_per_ru = np.split(action, self.num_rus)
        self.reward = 0
        cnt = 0
        completed = 0
        for ru_action in actions_per_ru:

            if self.validate_action(ru_action,cnt):
                completed += 1

                vnc,primary_path_idx,backup1_path_idx,backup2_path_idx,wavelength,request_type = ru_action

                DRCs = DRC_structure_T1()
                VNC = DRCs[vnc]

                primary_path = all_paths[rus[cnt]][primary_path_idx]
                backup1_path = all_paths[rus[cnt]][backup1_path_idx]
                backup2_path = all_paths[rus[cnt]][backup2_path_idx]

                node_activated=0

                node_activated+=self.update_state(primary_path,vnc,wavelength)
                node_activated+=self.update_state(backup1_path,vnc,wavelength)
                node_activated+=self.update_state(backup2_path,vnc,wavelength)


                wavelength_activation=1

                if wavelength in self.wavelengths :
                    wavelength_activation = 0
                else:
                    self.wavelengths.append(wavelength)

                revenue = 5

                # if request_type == 0:
                #     revenue = 10
                # elif request_type == 1:
                #     revenue = 5
                # else:
                #     revenue = 2.5
                

                centralization = (VNC.id+1)

                self.reward += centralization + revenue - wavelength_activation - node_activated
                info = {}
            else :
                self.reward = -2*(self.num_rus-completed)
                self.end_ep = True
                info = {"is_success": False}
                break
            cnt += 1

        self.node_cpu = []
    
        for i in crs:
            self.node_cpu.append(self.graph.get_node(i[0])["capacity"])
            
        self.link_bandwidth = []  
        self.link_latency = []
        self.link_wavelength = []

        for i in links:
            self.link_bandwidth.append(self.graph.get_edge(i[0],i[1])["bandwidth"])
            self.link_latency.append(self.graph.get_edge(i[0],i[1])["latency"])
            val = [0]*self.num_wavelengths
            for j in self.graph.get_edge(i[0],i[1])["values"]:
                val[j] = 1
            self.link_wavelength.append(val)
        
        self.state = np.concatenate(
            [self.node_cpu, self.link_bandwidth, self.link_latency, np.array(self.link_wavelength).flatten()], dtype=np.float32
        )

        
        return self.state, self.reward, self.end_ep,False, info
        # return self.graph,self.reward,self.end_ep,info

    
    def read_topology(self):
        """
      READ T1 TOPOLOGY FILE
      Implements the topology from reading the json file and creates the main structure that is used by the stages of the model.
      :rtype: Inserts topology data into global structures, so the method has no return
      """
        with open(self.links_file) as json_file:
            data = json.load(json_file)

            # Creates the set of links with delay and capacity read by the json file, stores the links in the global list "links"
            json_links = data["links"]
            for item in json_links:
                link = item
                source_node = link["fromNode"]
                destination_node = link["toNode"]

                # capacity[(source_node, destination_node)] = link["capacity"]
                # delay[(source_node, destination_node)] = link["delay"]
                links.append((source_node, destination_node,int(link["capacity"]) , int(link["delay"])))

            # Creates the set of CRs with RAM and CPU in a global list "crs" -- cr[0] is the network Core node
            # crs.append((0,0,0))
            with open(self.nodes_file) as json_file:
                data = json.load(json_file)
                json_nodes = data["nodes"]
                for item in json_nodes:
                    node = item
                    CR_id = int(node["nodeNumber"])
                    # node_RAM = node["RAM"]
                    node_CPU = int(node["cpu"])
                    rus = int(node["RU"])
                    cr = (CR_id, node_CPU, rus)
                    crs.append(cr)
            # crs[0] = CR(0, 0, 0, 0)

    def render(self):
        print(self.graph.nodes)
        print(self.graph.edges)
        pass

    def close(self):
        pass
    


# env = Sliavalilran()
# # state = env.reset()
# done = False

# actions = []


# for episodes in range(10):
#     rwd = 0
#     while not done:
#         vnc = random.randint(1,9)
#         path = random.randint(0,len(rus)-1)
#         wavelength = random.randint(1,10)
#         # print(wavelength)
        
#         action = {'VNC':vnc,
#                 'path':all_paths[rus[path]],
#                 'wavelength':wavelength
#                 }  # Random action for testing
#         # print(action)
#         state, reward, done,_ = env.step(action)
#         rwd += reward
#         # env.render()
#     print(rwd)
#     done = False
#     # print("reset")
#     env.reset()

# from stable_baselines3 import PPO
# from stable_baselines3.common.envs import DummyVecEnv

# # Create and wrap the environment
# env = Sliavalilran()
# env = DummyVecEnv([lambda: env])  # Vectorize the environment for PPO

# # Define the PPO model (using an MLP policy)
# model = PPO("MlpPolicy", env, verbose=1)

# # Train the model
# model.learn(total_timesteps=10)  # Adjust timesteps as per your needs

# # Save the trained model
# model.save("ppo_custom_env")

# # Load the trained model for evaluation
# model = PPO.load("ppo_custom_env")

# # Test the trained model
# obs = env.reset()
# for i in range(1000):
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()  # Optionally render the environment





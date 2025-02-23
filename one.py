'''
Feedback Neural Network
Rick Howell
the.rickhowell@gmail.com

This program is designed to implement 'feeling' into the classic neural network model
We use filters and oscillators to facilitate this
'''

import numpy as np
import cv2
import textwrap
import math
import time
from typing import List, Tuple, Dict

WINDOW_NAME = 'Feedback Neural Network'
NUM_AGENTS = 20
INIT_FOOD = 10
INIT_ENERGY = 100

main_width = 800
info_width = 200
height = 800

specimen_idx = 0

NUM_GENES = 42
GENE_LENGTH = 6
HIDDEN_LAYERS = 2
BASE = 36

FORCE_FACTOR = 0.5
MUTATION_RATE = 0.1

FILTERS = ['LP', 'HP']

class Agent:

    def __init__(self, hidden_layers: int = 2):
        self.base_color = np.random.randint(0, 255, 3)
        self.age = 0
        self.freq_main = 10 + np.random.normal(0, 10)
        self.energy = INIT_ENERGY + np.random.normal(0, 1)
        self.pos = (np.random.randint(0, main_width), np.random.randint(0, height))
        self.vel = (2 * np.random.rand() - 1, 2 * np.random.rand() - 1)
        self.genome = self.make_genome()

        self.time_since_birth = 0
        self.time_since_food = 0

        self.hidden_layers = hidden_layers

        self.input_nodes = ['age', 'velocity', 'near_food', 'energy', 'tsb', 'tsf', 'noise']
        self.output_nodes = ['vel_x', 'vel_y', 'vel', 'eat', 'mitosis', 'main_freq']
        self._hidden_nodes = ['identity', 'bias', 'invert', 'LP', 'HP', 'fm', 'am', 'noise']
        self.hidden_nodes = []

        self.z1_nodes = []

        for i in range(hidden_layers):
            hn = self._hidden_nodes.copy()
            for item in hn:
                item.join(str(i))
            self.hidden_nodes.extend(hn)

        self.connections = self._decode_genome(self.genome)

        # We'll go through the connections and count how many filters there are
        # Then add them to the z1_nodes
        for c in self.connections:
            self.z1_nodes.append(0)

    def __str__(self):
        return f'{self.age}, \n{self.connections}, \n{self.energy}, \n{self.freq_main}, \n{self.genome}'

    def make_genome(self):
        return ''.join(np.random.choice(list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'), NUM_GENES * GENE_LENGTH))
    
    def _decode_genome(self, genome: str) -> Dict[str, List[Tuple[str, float]]]:
        '''Decodes the genome into a dictionary of nodes and weights'''

        connections = {node: [] for node in self.input_nodes + self.output_nodes + self.hidden_nodes}

        for i in range(0, len(genome), GENE_LENGTH):
            if i + GENE_LENGTH <= len(genome):
                gene = genome[i:i+GENE_LENGTH]

                source_index = int(gene[0], BASE)
                dest_index = int(gene[1], BASE)
                layer = int(gene[2], BASE) % (self.hidden_layers + 1)
                weight = (int(gene[3:], BASE) / (BASE ** 3)) * 2.0 - 1.0

                if layer == 0:
                    source_index = source_index % len(self.input_nodes)
                    dest_index = dest_index % len(self.hidden_nodes)
                
                elif layer == self.hidden_layers:
                    source_index = source_index % len(self.hidden_nodes)
                    dest_index = dest_index % len(self.output_nodes)

                else:
                    source_index = source_index % len(self.hidden_nodes)
                    dest_index = dest_index % len(self.hidden_nodes)
                
                source_node = self.input_nodes[source_index] if layer == 0 else self.hidden_nodes[source_index]
                dest_node = self.hidden_nodes[dest_index] if layer < self.hidden_layers else self.output_nodes[dest_index]

                connections[source_node].append((dest_node, weight))

        return connections
    
    def process(self, inputs: Dict[str, float]) -> Dict[str, float]:
        node_values = {**inputs}

        # If the node values are not present, we'll initialize them to 0
        for node in self.hidden_nodes + self.output_nodes:
            if node not in node_values:
                node_values[node] = 0.0
        
        # Process hidden layers
        for _ in range(self.hidden_layers):
            z1_block = 0
            new_values = {}
            for node in self.hidden_nodes + self.output_nodes:
                incoming = sum(node_values[src] * weight for src, connections in self.connections.items() for dest, weight in connections if dest == node)
                if 'identity' in node:
                    new_values[node] = incoming
                elif 'bias' in node:
                    new_values[node] = incoming + 0.5
                elif 'invert' in node:
                    new_values[node] = -incoming
                elif 'LP' in node:
                    new_values[node] = self.z1_nodes[z1_block] + 0.1 * (incoming - self.z1_nodes[z1_block])
                    self.z1_nodes[z1_block] = new_values[node]
                    z1_block += 1
                elif 'HP' in node:
                    new_values[node] = 0.1 * (incoming - self.z1_nodes[z1_block]) - self.z1_nodes[z1_block] 
                    self.z1_nodes[z1_block] = new_values[node]
                    z1_block += 1
                elif 'fm' in node:
                    new_values[node] = np.sin(2.0 * np.pi * time.time() * 2000 * incoming * self.freq_main)
                elif 'am' in node:
                    new_values[node] = incoming * np.sin(2.0 * np.pi * self.freq_main * time.time())
                elif 'noise' in node:
                    new_values[node] = np.random.normal(0, abs(incoming))

            node_values.update(new_values)
        
        # Process output layer
        outputs = {}
        for node in self.output_nodes:
            incoming = sum(node_values[src] * weight for src, connections in self.connections.items() for dest, weight in connections if dest == node)
            outputs[node] = np.tanh(incoming)  # Using tanh activation for output nodes
        
        return outputs
    
    def update(self, inputs: Dict[str, float]):
        for k, v in inputs.items():

            if math.isnan(v):
                break

            if k == 'vel_x':
                self.vel = (self.vel[0] + v, self.vel[1])
            elif k == 'vel_y':
                self.vel = (self.vel[0], self.vel[1] + v)
            elif k == 'vel':
                self.vel = (self.vel[0] + v, self.vel[1] + v)
            elif k == 'main_freq':
                self.freq_main += v
                self.freq_main = abs(self.freq_main)
            elif k == 'eat':
                # find closest piece of food
                idx = 0
                min = 1000
                for i in range(len(food_positions)):
                    distance = np.linalg.norm(np.subtract(food_positions[i], self.pos))
                    if distance < min:
                        min = distance
                        idx = i
                # try to eat it
                if min < 10 and v > 0:
                    self.energy += food_energy[idx]
                    food_energy.pop(idx)
                    food_positions.pop(idx)
                    self.time_since_food = 0
            elif k == 'mitosis':
                if v > 0 and self.age > np.random.normal(100, 25):
                    agents.append(self.mitosis())
            
        self.vel = np.clip(self.vel, -1.0, 1.0)
        self.age += 0.1
        self.energy -= 0.01

        self.energy = max(0.0, self.energy)
        self.freq_main = 10 * np.tanh(self.freq_main / 10)

        self.time_since_food += time.time()
        self.time_since_birth += time.time()

    def move(self):
        self.pos += np.multiply(FORCE_FACTOR, self.vel)
        self.pos = np.mod(self.pos, (main_width, height))

    def get_color(self):
        ef = np.multiply(self.base_color, self.energy / INIT_ENERGY)
        ef = np.clip(ef, 0, 255)
        return ef

    def is_alive(self) -> bool:
        return self.energy > 0
    
    def mutate_genome(self) -> str:
        # flip a few bits to mutate :)
        new_genome = ''
        for i in range(0, len(self.genome), GENE_LENGTH):
            if i + GENE_LENGTH <= len(self.genome):
                if np.random.rand() < MUTATION_RATE:
                    new_gene = np.random.choice(list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'), GENE_LENGTH)
                else:
                    new_gene = self.genome[i:i+GENE_LENGTH]

                new_genome += ''.join(new_gene)
        print(new_genome)
        return new_genome
    
    def mitosis(self) -> 'Agent':
        # Make a new agent with a mutated genome
        new_energy = np.maximum(1.0, self.energy * np.random.normal(0.5, 0.1))
        self.energy = new_energy
        new_genome = self.mutate_genome()
        new_agent = Agent(HIDDEN_LAYERS)
        new_agent.energy = new_energy
        gaus_pos = np.random.normal(0, 10, 2)
        new_agent.pos = (self.pos[0] + gaus_pos[0], self.pos[1] + gaus_pos[1])
        new_agent.genome = new_genome
        print('New Genome!', new_agent.genome)
        new_agent.hidden_nodes = []
        new_agent.z1_nodes = []
        for i in range(new_agent.hidden_layers):
            hn = new_agent._hidden_nodes.copy()
            for item in hn:
                item.join(str(i))
            new_agent.hidden_nodes.extend(hn)
        new_agent.connections = new_agent._decode_genome(new_agent.genome)
        for c in new_agent.connections:
            new_agent.z1_nodes.append(0)
        new_agent.base_color = self.base_color

        self.age = 0
        self.genome = self.mutate_genome()
        self.hidden_nodes = []
        self.z1_nodes = []
        for i in range(self.hidden_layers):
            hn = self._hidden_nodes.copy()
            for item in hn:
                item.join(str(i))
            self.hidden_nodes.extend(hn)
        self.connections = self._decode_genome(self.genome)
        for c in self.connections:
            self.z1_nodes.append(0)

        return new_agent


# Let's get some agents working
agents = []
for _ in range(NUM_AGENTS):
    agents.append(Agent())

map = np.zeros((main_width, height, 3), dtype=np.uint8)

# We'll populate the map with food:
food_positions = []
food_energy = []

def add_food(N: int = 1):
    for _ in range(N):
        # get a random position
        pos = (int(np.random.rand() * main_width), int(np.random.rand() * height))
        food_positions.append(pos)
        food_energy.append(1)



def get_food_dist(a: Agent) -> float:
    # we'll sum all the distances from food positions to the agent
    sum = 0.0
    for f in food_positions:
        sum += np.linalg.norm((f[0] - a.pos[0], f[1] - a.pos[1]))

    if len(food_positions) < 1:
        return sum
    
    return sum / (len(food_positions) * 700)

def get_inputs(a: Agent) -> Dict[str, float]:
    inputs = {
        'age': a.age * np.sin(2 * np.pi * a.freq_main * time.time()),
        'velocity': np.linalg.norm(a.vel) * np.sin(2 * np.pi * a.freq_main * time.time()),
        'near_food': get_food_dist(a), 
        'energy': a.energy * np.sin(2 * np.pi * a.freq_main * time.time()),
        'tsb': a.time_since_birth * np.sin(2 * np.pi * a.freq_main * time.time()),
        'tsf': a.time_since_food * np.sin(2 * np.pi * a.freq_main * time.time()),
        'noise': 2.0 * np.random.rand() - 1.0
    }
    return inputs

def draw():
    # map = np.zeros((main_width, height, 3), dtype=np.uint8)
    map = np.full((main_width, height, 3), (80, 40, 10), dtype=np.uint8)
    # draw food
    for f in food_positions:
        cv2.circle(map, f, 3, (120, 255, 180), -1, lineType=cv2.LINE_AA)

    # draw agents
    idx = 0
    for a in agents:
        x = a.pos[0]
        x = np.round(x)
        x = np.clip(x, 0, main_width - 1)
        x = int(x)

        y = a.pos[1]
        y = np.round(y)
        y = np.clip(y, 0, height - 1)
        y = int(y)

        cv2.circle(map, (x, y), 13, a.get_color(), 4, lineType=cv2.LINE_AA)

        vx = int(a.vel[0] * 13)
        vy = int(a.vel[1] * 13)
        cv2.circle(map, (x + vx, y + vy), 3, a.get_color(), -1, lineType=cv2.LINE_AA)

        if idx == specimen_idx:
            cv2.circle(map, (x, y), 20, (240, 240, 255), 2, lineType=cv2.LINE_AA)
        
        idx += 1

    return map



def update():
    global agents, specimen_idx
    agents = [a for a in agents if a.is_alive()]
    specimen_idx = np.mod(specimen_idx, len(agents))
    for a in agents:
        a.move()
        a.update(a.process(get_inputs(a)))
        



####################################################
                    # INFO #
####################################################

font = cv2.FONT_HERSHEY_PLAIN
info = np.full((height, info_width, 3), (100, 100, 100), dtype=np.uint8)

# let's first try to draw the network connections:

spacing = 20


def draw_network(specimen: Agent, info_surface: np.ndarray, start_y: int = 100):
    """Draws the neural network based on genome encoding"""
    # Calculate x positions for each layer
    layer_height = (height - 100) // (specimen.hidden_layers + 1)
    
    # We'll first draw the nodes
    # Then we'll populate with edges
    draw_info = []

    for n in range(len(specimen.input_nodes)):
        pos = (20 + n * info_width // len(specimen.input_nodes), start_y)
        cv2.circle(info_surface, pos, 3, (255, 255, 255), -1)
        if specimen.connections.get(specimen.input_nodes[n]):
            dest = specimen.connections.get(specimen.input_nodes[n])
            


def update_info():
    global info

    specimen = agents[specimen_idx]

    info = np.full((height, info_width, 3), (20, 20, 20), dtype=np.uint8)  # Darker background
    
    # Draw basic stats
    cv2.putText(info, 'Age:     ' + str(round(specimen.age, 2)), (10, 20), font, 1, (255, 255, 255))
    cv2.putText(info, 'Energy:  ' + str(round(specimen.energy, 2)), (10, 40), font, 1, (255, 255, 255))
    cv2.putText(info, 'Freq:    ' + str(round(specimen.freq_main, 2)), (10, 60), font, 1, (255, 255, 255))
    cv2.putText(info, 'Velocity:' + str(np.round(specimen.vel, 2)), (10, 80), font, 1, (255, 255, 255))
    # Draw genome info

    cv2.putText(info, f'Genome:', (10, 100), font, 1, (255, 255, 255))
    wrap_text = textwrap.wrap(specimen.genome, 6)
    space = 1
    for i in enumerate(wrap_text):
        cv2.putText(info, str(i), (10, 13 * (space) + 100), font, 1, (255, 255, 255))
        space += 1

    # Draw the network
    # draw_network(specimen, info, 120)

################ MAIN LOOP #########################
# ================================================ #
####################################################

add_food(36)

testagent = Agent()
print(testagent)

img = None

print(get_food_dist(testagent))

while True:
    # refresh
    img = draw()

    display = np.concatenate((img, info), axis=1)

    update()
    update_info()

    # Show the image
    cv2.imshow(WINDOW_NAME, display)


    # Check if the user pressed a key
    key = cv2.waitKey(1)
    if key == ord('q') or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break
    if key == ord('d'):
        specimen_idx += 1
        specimen_idx = np.mod(specimen_idx, NUM_AGENTS)
    if key == ord('a'):
        specimen_idx -= 1
        specimen_idx = np.mod(specimen_idx, NUM_AGENTS)

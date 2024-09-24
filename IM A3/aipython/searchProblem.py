# searchProblem.py - representations of search problems
# AIFCA Python3 code Version 0.9.5 Documentation at http://aipython.org
# Download the zip file and read aipython.pdf for documentation

# Artificial Intelligence: Foundations of Computational Agents http://artint.info
# Copyright David L Poole and Alan K Mackworth 2017-2022.
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: http://creativecommons.org/licenses/by-nc-sa/4.0/deed.en

import numpy as np
import random

class Search_problem(object):
    """A search problem consists of:
    * a start node
    * a neighbors function that gives the neighbors of a node
    * a specification of a goal
    * a (optional) heuristic function.
    The methods must be overridden to define a search problem."""

    def start_node(self):
        """returns start node"""
        raise NotImplementedError("start_node")   # abstract method
    
    def is_goal(self,node):
        """is True if node is a goal"""
        raise NotImplementedError("is_goal")   # abstract method

    def neighbors(self,node):
        """returns a list of the arcs for the neighbors of node"""
        raise NotImplementedError("neighbors")   # abstract method

    def heuristic(self,n):
        """Gives the heuristic value of node n.
        Returns 0 if not overridden."""
        return 0

class Arc(object):
    """An arc has a from_node and a to_node node and a (non-negative) cost"""
    def __init__(self, from_node, to_node, cost=1, action=None):
        assert cost >= 0, ("Cost cannot be negative for"+
                           str(from_node)+"->"+str(to_node)+", cost: "+str(cost))
        self.from_node = from_node
        self.to_node = to_node
        self.action = action
        self.cost=cost

    def __repr__(self):
        """string representation of an arc"""
        if self.action:
            return str(self.from_node)+" --"+str(self.action)+"--> "+str(self.to_node)
        else:
            return str(self.from_node)+" --> "+str(self.to_node)

class Search_problem_from_explicit_graph(Search_problem):
    """A search problem consists of:
    * a list or set of nodes
    * a list or set of arcs
    * a start node
    * a list or set of goal nodes
    * a dictionary that maps each node into its heuristic value.
    * a dictionary that maps each node into its (x,y) position
    """

    def __init__(self, nodes, arcs, start=None, goals=set(), hmap={}, positions={}):
        self.neighs = {}
        self.nodes = nodes
        for node in nodes:
            self.neighs[node]=[]
        self.arcs = arcs
        for arc in arcs:
            self.neighs[arc.from_node].append(arc)
        self.start = start
        self.goals = goals
        self.hmap = hmap
        self.positions = positions

    def start_node(self):
        """returns start node"""
        return self.start
    
    def is_goal(self,node):
        """is True if node is a goal"""
        return node in self.goals

    def neighbors(self,node):
        """returns the neighbors of node"""
        return self.neighs[node]

    def heuristic(self,node):
        """Gives the heuristic value of node n.
        Returns 0 if not overridden in the hmap."""
        if node in self.hmap:
            return self.hmap[node]
        else:
            return 0
        
    def __repr__(self):
        """returns a string representation of the search problem"""
        res=""
        for arc in self.arcs:
            res += str(arc)+".  "
        return res

    def neighbor_nodes(self,node):
        """returns an iterator over the neighbors of node"""
        return (path.to_node for path in self.neighs[node])

class Path(object):
    """A path is either a node or a path followed by an arc"""
    
    def __init__(self,initial,arc=None):
        """initial is either a node (in which case arc is None) or
        a path (in which case arc is an object of type Arc)"""
        self.initial = initial
        self.arc=arc
        if arc is None:
            self.cost=0
        else:
            self.cost = initial.cost+arc.cost

    def end(self):
        """returns the node at the end of the path"""
        if self.arc is None:
            return self.initial
        else:
            return self.arc.to_node

    def nodes(self):
        """enumerates the nodes for the path.
        This starts at the end and enumerates nodes in the path backwards."""
        current = self
        while current.arc is not None:
            yield current.arc.to_node
            current = current.initial
        yield current.initial

    def initial_nodes(self):
        """enumerates the nodes for the path before the end node.
        This starts at the end and enumerates nodes in the path backwards."""
        if self.arc is not None:
            yield from self.initial.nodes()
        
    def __repr__(self):
        """returns a string representation of a path"""
        if self.arc is None:
            return str(self.initial)
        elif self.arc.action:
            return (str(self.initial)+"\n   --"+str(self.arc.action)
                    +"--> "+str(self.arc.to_node))
        else:
            return str(self.initial)+" --> "+str(self.arc.to_node)


# Here is the start of my code implementations
# It takes maze data and transforms it into a format usable with the above functions

# A node class to initialize Node objects with a number and name
class Node():

    def __init__(self,name,number):
        self.number = number
        self.name = name

# A function to get the raw maze data and the possible starting positions
def getData():
    # A random connectivity percentage and maze number are chosen
    connectivity = random.randrange(1,4)
    connectivityPercentages = ['0','30','60','100']
    maze = random.randrange(1,10)
    # Required data is loaded in
    mazeData = np.loadtxt(
        'SCMP/SCMP' + str(connectivity) + '/mazes/m' + connectivityPercentages[connectivity - 1] + '_' + str(
        maze) + '.mz')
    starts = np.loadtxt('SCMP/SCMP' + str(connectivity) + '/starting_locations.loc')
    # Print statements for user
    print("Maze connectivity is: " + connectivityPercentages[connectivity - 1] + "%")
    print("Maze number selected: " + str(maze))
    # Convert data to integers and remove first 3 data points from each array
    mazeData = mazeData.astype(int)
    starts = starts.astype(int)
    mazeData = mazeData[3:]
    starts = starts[3:]
    return mazeData,starts

# A function to get a randomly assigned starting node position from the raw data
def getStartNode(starts):
    # Pick a random list index
    s = random.choice(range(len(starts)))
    # If the index is an even number
    if s % 2 == 0:
        # It has selected a column value index, so the corresponding row value index is the one right after it
        s2 = s + 1
        col = starts[s]
        row = starts[s2]
        # Get the node's name and return it
        start = letters[row] + str(col)
        return start
    else:
        # It has selected a row value index, so the corresponding column value is the one preceding it.
        s0 = s - 1
        col = starts[s0]
        row = starts[s]
        # Get the node's name and return it
        start = letters[row] + str(col)
        return start

# A function to return a list of Node objects
def getNodes(mazeData,letters):
    lCount = 0
    count = 0
    nodes = []
    # For each data-point in the maze data
    for i in mazeData:
        # Construct a name for this node
        name = letters[lCount] + str(count)
        # Create a Node object and append it to the list
        nodes.append(Node(name, i))
        count += 1
        # A system to ensure the naming conventions are correct
        if count >= 15:
            lCount += 1
            count = 0
    return nodes

# A function to return all the node names in a set
def getNodeNames(nodes):
    nodeNames = set()
    # Get the name from each node object and add it to the set
    for node in nodes:
        nodeNames.add(node.name)
    return nodeNames

# A function to return a list of every single arc in the maze
def getArcs(nodes,letters,walls):
    arcs = []
    # For every node object in the maze
    for n in nodes:
        # Get the node's name and number/data
        na = n.name
        nu = n.number
        l = letters.index(na[0])
        # If the node is not the goal node
        if nu < 16:
            # Get the bit value that indicates the walls surrounding that node
            w = walls[nu]
        # If the node is the goal node
        else:
            # Assign it to a variable, then get the relevant bit value that indicates the walls surrounding the node
            g = na
            nu = nu - 16
            w = walls[nu]
        # If the node has no left wall and isn't in the leftmost column
        if w[1] == '0' and na[1:] != '0':
            for i in nodes:
                # Find the node to the initial node's left and get its name and number
                if i.name == na[0] + str(int(na[1:]) - 1):
                    nu2 = i.number
                    na2 = i.name
            # If this second node is the goal node, ensure correct bit value is retrieved
            if nu2 > 15:
                nu2 -= 16
            w2 = walls[nu2]
            # If the second node has no right wall/there is a direct path between the two nodes
            if w2[3] == '0':
                # Create an arc between the two nodes and append it to the arc list
                arcs.append(Arc(na, na2, 1))
        # If the node has no bottom wall and isn't on the bottom row
        if w[2] == '0' and na[0] != 'o':
            for i in nodes:
                # Find the node directly below the initial node and get its name and number
                if i.name == letters[l + 1] + na[1:]:
                    nu2 = i.number
                    na2 = i.name
            # If this second node is the goal node, ensure correct bit value is retrieved
            if nu2 > 15:
                nu2 -= 16
            w2 = walls[nu2]
            # If the second node has no top wall/there is a direct path between the two nodes
            if w2[4] == '0':
                # Create an arc between the two nodes and append it to the arc list
                arcs.append(Arc(na, na2, 1))
        # If the node has no right wall and isn't in the rightmost column
        if w[3] == '0' and na[1:] != '14':
            for i in nodes:
                # Find the node to the initial node's right and get its name and number
                if i.name == na[0] + str(int(na[1:]) + 1):
                    nu2 = i.number
                    na2 = i.name
            # If this second node is the goal node, ensure correct bit value is retrieved
            if nu2 > 15:
                nu2 -= 16
            w2 = walls[nu2]
            # If the second node has no left wall/there is a direct path between the two nodes
            if w2[1] == '0':
                # Create an arc between the two nodes and append it to the arc list
                arcs.append(Arc(na, na2, 1))
        # If the node has no top wall and isn't on the top row
        if w[4] == '0' and na[0] != 'a':
            for i in nodes:
                # Find the node directly above the initial node and get its name and number
                if i.name == letters[l - 1] + na[1:]:
                    nu2 = i.number
                    na2 = i.name
            # If this second node is the goal node, ensure correct bit value is retrieved
            if nu2 > 15:
                nu2 -= 16
            w2 = walls[nu2]
            # If the second node has no bottom wall/there is a direct path between the two nodes
            if w2[2] == '0':
                # Create an arc between the two nodes and append it to the arc list
                arcs.append(Arc(na, na2, 1))
    return arcs,g

# A heuristic function for A* searches
def hFunction(nodeNames,positions,g):
    hmap = {}
    # For each node in the maze
    for n in nodeNames:
        # Get the position of both the current node and the goal node
        currentNode = positions.get(n)
        goalNode = positions.get(g)
        # If the goal node is higher in the maze
        if currentNode[0] > goalNode[0]:
            h1 = int(currentNode[0]) - int(goalNode[0])
        # If the goal node is lower in the maze
        else:
            h1 = int(goalNode[0]) - int(currentNode[0])
        # If the goal node is further left in the maze
        if currentNode[1] > goalNode[1]:
            h2 = currentNode[1] - goalNode[1]
        # If the goal node is further right in the maze
        else:
            h2 = goalNode[1] - currentNode[1]
        # Add the difference between the two positions and multiply by 2
        h = (h1 + h2)
        hmap[n] = h
    return hmap

# A function the defines the positions of nodes in the maze
def getPositions(nodeNames,letters):
    positions = {}
    for name in nodeNames:
        # Save the node name and position as a dictionary key-value pair
        positions[name] = (letters.index(name[0]), int(name[1:]))
    return positions

# Get maze data
mazeData,starts = getData()
# Create arrays for row values and possible wall bit values
letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o']
walls = ['00000', '00001', '00010', '00011', '00100', '00101', '00110', '00111', '01000', '01001', '01010', '01011',
         '01100', '01101', '01110', '01111']
# Get starting node
start = getStartNode(starts)
# Create node objects
nodes = getNodes(mazeData,letters)
# Get the names of all the nodes
nodeNames = getNodeNames(nodes)
# Get all maze arcs and the goal node
arcs,g = getArcs(nodes,letters,walls)
# Get all node positions in the maze
positions = getPositions(nodeNames,letters)
# Apply heuristic function to all nodes
hmap = hFunction(nodeNames,positions,g)
print("Starting node: "+start)
goal = set()
goal.add(g)
print("Goal node: "+list(goal)[0])

DFSTestProblem = Search_problem_from_explicit_graph(nodeNames,arcs,start=start,goals=goal,positions=positions)
ASTestProblem = Search_problem_from_explicit_graph(nodeNames,arcs,start=start,goals=goal,hmap=hmap)
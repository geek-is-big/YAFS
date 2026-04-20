# -*- coding: utf-8 -*-
import logging


import networkx as nx
import warnings


class Topology:
    """
    This class unifies the functions to deal with **Complex Networks** as a network topology within of the simulator. In addition, it facilitates its creation, and assignment of attributes.
    """

    LINK_BW = "BW"
    "Link feature: Bandwidth"

    LINK_PR = "PR"
    "Link feauture:  Propagation delay"

    LINK_RTR = "RTR"
    "Link feature: Retransmission rate in range [0,1]"

    # LINK_LATENCY = "LATENCY"
    # " A edge or a network link has a Bandwidth"

    NODE_IPT = "IPT"
    "Node feature: IPS . Instructions per Simulation Time "



    def __init__(self, logger=None):

        # G is a nx.networkx graph
        self.G = None
        self.nodeAttributes = {}
        self.logger = logger or logging.getLogger(__name__)




    def __init_uptimes(self):
        for key in self.nodeAttributes:
            self.nodeAttributes[key]["uptime"] = (0, None)

    def get_edges(self):
        """
        Returns:
            list: a list of graph edges, i.e.: ((1,0),(0,2),...)
        """
        return self.G.edges

    def get_edge(self,key):
        """
        Args:
            key (str): a edge identifier, i.e. (1,9)

        Returns:
            list: a list of edge attributes
        """
        return self.G.edges[key]

    def get_nodes(self):
        """
        Returns:
            list: a list of all nodes features
        """
        return self.G.nodes

    def get_node(self, key):
        """
        Args:
            key (int): a node identifier

        Returns:
            list: a list of node features
        """
        return self.G.node[key]


    def get_info(self):
        return self.nodeAttributes

    def create_topology_from_graph(self, G):
        """
        It generates the topology from a NetworkX graph

        Args:
             G (*networkx.classes.graph.Graph*)
        """
        if isinstance(G, nx.classes.graph.Graph):
            self.G = G
        else:
            raise TypeError

    def create_random_topology(self, nxGraphGenerator, params):
        """
        It generates the topology from a Graph generators of NetworkX

        Args:
             nxGraphGenerator (function): a graph generator function

        Kwargs:
            params (dict): a list of parameters of *nxGraphGenerator* function
        """
        try:
            self.G = nxGraphGenerator(*params)
        except:
            raise Exception

    def load(self, data):
        warnings.warn("The load function will merged with load_all_node_attr function",
                      FutureWarning,
                      stacklevel=8
                      )
        """
            It generates the topology from a JSON file
            see project example: Tutorial_JSONModelling

            Args:
                 data (str): a json
        """
        self.G = nx.Graph()
        for edge in data["link"]:
            rtr = min(max(float(edge.get(self.LINK_RTR, 0.0)), 0.0), 1.0)
            self.G.add_edge(
                edge["s"],
                edge["d"],
                BW=edge[self.LINK_BW],
                PR=edge[self.LINK_PR],
                RTR=rtr,
            )

        # Adding custom WATT LINK attributes if exist
        valuesWTrans = {}
        for edge in data["link"]:
            try:
                # register link consumption depending on direction since YAFS uses undirected graphs
                nx.set_edge_attributes(self.G,
                                       values={(edge["s"],edge["d"]): edge["WATT_TRANS"]},
                                       name=f"WATT_TRANS_{edge['s']}-{edge['d']}")
                 
                # preserve default WATT attributes values, in case no directed values are specified
                valuesWTrans[(edge["s"],edge["d"])] = edge["WATT_TRANS"]
            except KeyError:
                valuesWTrans[(edge["s"],edge["d"])] = 0.0
        nx.set_edge_attributes(self.G, values=valuesWTrans, name="WATT_TRANS")

        valuesWRecv = {}
        for edge in data["link"]:
            try:
                # register link consumption depending on direction since YAFS uses undirected graphs
                nx.set_edge_attributes(self.G,
                                       values={(edge["s"],edge["d"]): edge["WATT_RECV"]},
                                       name=f"WATT_RECV_{edge['s']}-{edge['d']}")

                # preserve default WATT attributes values, in case no directed values are specified
                valuesWRecv[(edge["s"],edge["d"])] = edge["WATT_RECV"]
            except KeyError:
                    valuesWRecv[(edge["s"],edge["d"])] = valuesWTrans[(edge["s"],edge["d"])]
        nx.set_edge_attributes(self.G, values=valuesWRecv, name="WATT_RECV")

        # Optional directed tail-energy attributes (Wh per transfer event)
        for edge in data["link"]:
            for key, name in (("TAIL_TRANS", "TAIL_TRANS"), ("TAIL_RECV", "TAIL_RECV")):
                src = edge["s"]
                dst = edge["d"]
                directed = edge.get(f"{key}_{src}-{dst}", edge.get(key, 0.0))
                try:
                    directed = float(directed)
                except (TypeError, ValueError):
                    directed = 0.0
                nx.set_edge_attributes(
                    self.G,
                    values={(src, dst): directed},
                    name=f"{name}_{src}-{dst}",
                )
            # Preserve any explicitly provided directed edge attributes.
            for attr_name, attr_val in edge.items():
                if attr_name.startswith("WATT_TRANS_") or attr_name.startswith("WATT_RECV_") \
                        or attr_name.startswith("TAIL_TRANS_") or attr_name.startswith("TAIL_RECV_"):
                    try:
                        attr_val = float(attr_val)
                    except (TypeError, ValueError):
                        attr_val = 0.0
                    nx.set_edge_attributes(
                        self.G,
                        values={(edge["s"], edge["d"]): attr_val},
                        name=attr_name,
                    )


        #TODO This part can be removed in next versions
        for node in data["entity"]:
            self.nodeAttributes[node["id"]] = node
        #end remove

        # Correct way to use custom and mandatory topology attributes

        valuesIPT = {}
        # valuesRAM = {}
        for node in data["entity"]:
            try:
                valuesIPT[node["id"]] = node["IPT"]
            except KeyError:
                valuesIPT[node["id"]] = 0
            # try:
            #     valuesRAM[node["id"]] = node["RAM"]
            # except KeyError:
            #     valuesRAM[node["id"]] = 0


        nx.set_node_attributes(self.G,values=valuesIPT,name="IPT")
        # nx.set_node_attributes(self.G,values=valuesRAM,name="RAM")

        self.__init_uptimes()

    def load_all_node_attr(self,data):
        self.G = nx.Graph()
        for edge in data["link"]:
            rtr = min(max(float(edge.get(self.LINK_RTR, 0.0)), 0.0), 1.0)
            self.G.add_edge(
                edge["s"],
                edge["d"],
                BW=edge[self.LINK_BW],
                PR=edge[self.LINK_PR],
                RTR=rtr,
            )
            src = edge["s"]
            dst = edge["d"]
            if "WATT_TRANS" in edge:
                nx.set_edge_attributes(
                    self.G,
                    values={(src, dst): edge["WATT_TRANS"]},
                    name=f"WATT_TRANS_{src}-{dst}",
                )
                nx.set_edge_attributes(
                    self.G,
                    values={(src, dst): edge["WATT_TRANS"]},
                    name="WATT_TRANS",
                )
            if "WATT_RECV" in edge:
                nx.set_edge_attributes(
                    self.G,
                    values={(src, dst): edge["WATT_RECV"]},
                    name=f"WATT_RECV_{src}-{dst}",
                )
                nx.set_edge_attributes(
                    self.G,
                    values={(src, dst): edge["WATT_RECV"]},
                    name="WATT_RECV",
                )
            for key, name in (("TAIL_TRANS", "TAIL_TRANS"), ("TAIL_RECV", "TAIL_RECV")):
                directed = edge.get(f"{key}_{src}-{dst}", edge.get(key, 0.0))
                try:
                    directed = float(directed)
                except (TypeError, ValueError):
                    directed = 0.0
                nx.set_edge_attributes(
                    self.G,
                    values={(src, dst): directed},
                    name=f"{name}_{src}-{dst}",
                )
            for attr_name, attr_val in edge.items():
                if attr_name.startswith("WATT_TRANS_") or attr_name.startswith("WATT_RECV_") \
                        or attr_name.startswith("TAIL_TRANS_") or attr_name.startswith("TAIL_RECV_"):
                    try:
                        attr_val = float(attr_val)
                    except (TypeError, ValueError):
                        attr_val = 0.0
                    nx.set_edge_attributes(
                        self.G,
                        values={(src, dst): attr_val},
                        name=attr_name,
                    )

        dc = {str(x): {} for x in data["entity"][0].keys()}
        for ent in data["entity"]:
            for key in ent.keys():
                dc[key][ent["id"]] = ent[key]
        for x in data["entity"][0].keys():
            nx.set_node_attributes(self.G, values=dc[x], name=str(x))

        for node in data["entity"]:
            self.nodeAttributes[node["id"]] = node

        self.__idNode = len(self.G.nodes)
        self.__init_uptimes()




    def load_graphml(self,filename):
        warnings.warn("The load_graphml function is deprecated and "
                      "will be removed in version 2.0.0. "
                      "Use NX.READ_GRAPHML function instead.",
                      FutureWarning,
                      stacklevel=8
                      )

        self.G = nx.read_graphml(filename)
        attEdges = {}
        for k in self.G.edges():
            attEdges[k] = {"BW": 1, "PR": 1}
        nx.set_edge_attributes(self.G, values=attEdges)
        attNodes = {}
        for k in self.G.nodes():
            attNodes[k] = {"IPT": 1}
        nx.set_node_attributes(self.G, values=attNodes)
        for k in self.G.nodes():
            self.nodeAttributes[k] = self.G.node[k] #it has "id" att. TODO IMPROVE


    def get_nodes_att(self):
        """
        Returns:
            A dictionary with the features of the nodes
        """
        return self.nodeAttributes

    def find_IDs(self,value):
        """
        Search for nodes with the same attributes that value

        Args:
             value (dict). example value = {"model": "m-"}. Only one key is admitted

        Returns:
            A list with the ID of each node that have the same attribute that the value.value
        """
        keyS = list(value.keys())[0]

        result = []
        for key in self.nodeAttributes.keys():
            val = self.nodeAttributes[key]
            if keyS in val:
                if value[keyS] == val[keyS]:
                    result.append(key)
        return result


    def size(self):
        """
        Returns:
            an int with the number of nodes
        """
        return len(self.G.nodes)

    def add_node(self, nodes, edges=None):
        """
        Add a list of nodes in the topology

        Args:
            nodes (list): a list of identifiers

            edges (list): a list of destination edges
        """
        self.__idNode = + 1
        self.G.add_node(self.__idNode)
        self.G.add_edges_from(zip(nodes, [self.__idNode] * len(nodes)))

        return self.__idNode

    def remove_node(self, id_node):
        """
        Remove a node of the topology

        Args:
            id_node (int): node identifier
        """

        self.G.remove_node(id_node)
        return self.size()

import pandas as pd
import numpy as np
from pathlib import Path

from yafs.metrics import Metrics


class Stats:

    def __init__(self,defaultPath="result"):
        self.df_link = pd.read_csv(defaultPath + "_link.csv")
        self.df = pd.read_csv(defaultPath + ".csv")
        self.df_offloading = None
        base = Path(defaultPath)
        name = base.name
        if name.startswith("sim_trace_"):
            tag = name[len("sim_trace_"):]
            # TODO: should the offloading_path be checked under the if name.startswith("sim_trace_"): ?
            offloading_path = base.with_name(f"offloading_decisions_{tag}.csv")
            if offloading_path.exists():
                self.df_offloading = pd.read_csv(offloading_path)


    def bytes_transmitted(self):
        return self.df_link["size"].sum()

    def count_messages(self):
        return len(self.df_link)


    def utilization(self,id_entity, total_time, from_time=0.0):
        if "time_service" not in self.df.columns: #cached
            self.df["time_service"] = self.df.time_out - self.df.time_in
        values = self.df.groupby("DES.dst").time_service.agg("sum")
        return values[id_entity] / total_time

    def compute_times_df(self):
        self.df["time_latency"] = self.df["time_reception"] - self.df["time_emit"]
        self.df["time_wait"] = self.df["time_in"] - self.df["time_reception"]  #
        self.df["time_service"] = self.df["time_out"] - self.df["time_in"]
        self.df["time_response"] = self.df["time_out"] - self.df["time_reception"]
        self.df["time_total_response"] = self.df["time_response"] + self.df["time_latency"]

    def times(self,time,value="mean"):
        if "time_response" not in self.df.columns:
            self.compute_times_df()
        return self.df.groupby("message").agg({time:value})



    def average_loop_response(self,time_loops):
        """
        No hay chequeo de la existencia del loop: user responsability
        """
        if "time_response" not in self.df.columns:
            self.compute_times_df()

        resp_msg = self.df.groupby("message").agg({"time_total_response": ["mean","count"]}) #Its not necessary to have "count"
        resp_msg.columns = ['_'.join(col).strip() for col in resp_msg.columns.values]
        results = []

        for loop in time_loops:
            total = 0.0
            for msg in loop:
                try:
                    total += resp_msg[resp_msg.index == msg].time_total_response_mean[0]
                except IndexError:
                    total +=0

            results.append(total)

        return results

    def get_watt(self,totaltime,topology,by=Metrics.WATT_SERVICE):
        results = {}
        nodeInfo = topology.get_info()
        if by == Metrics.WATT_SERVICE:
            # Tiempo de actividad / runeo
            if "time_response" not in self.df.columns:  # cached
                self.compute_times_df()

            nodes = self.df.groupby("TOPO.dst").agg({"time_service": "sum"})
            for id_node in nodes.index:
                results[id_node] = {"model": nodeInfo[id_node]["model"], "type": nodeInfo[id_node]["type"],
                                 "watt": nodes.loc[id_node].time_service * nodeInfo[id_node]["WATT"]}
        elif by == Metrics.WATT_LINK:
            linkInfo = topology.get_edges()
            df_link_grouped = self.df_link.groupby(["src","dst"]).agg({"latency":"sum"})
            direct_link_energy_enabled = {"tx_wh", "rx_wh"}.issubset(set(self.df_link.columns))
            if direct_link_energy_enabled:
                df_energy_grouped = self.df_link.groupby(["src","dst"]).agg({"tx_wh":"sum", "rx_wh":"sum"})
            tail_enabled = {"tail_tx_wh", "tail_rx_wh"}.issubset(set(self.df_link.columns))
            if tail_enabled:
                df_tail_grouped = self.df_link.groupby(["src","dst"]).agg({"tail_tx_wh":"sum", "tail_rx_wh":"sum"})
            for link in df_link_grouped.index:
                src = link[0]
                dst = link[1]
                if direct_link_energy_enabled:
                    energy_trans = float(df_energy_grouped.loc[link].tx_wh)
                    energy_recv = float(df_energy_grouped.loc[link].rx_wh)
                else:
                    time_link = df_link_grouped.loc[link].latency
                    try:
                        watt_trans = linkInfo[(src,dst)][f"WATT_TRANS_{src}-{dst}"]
                    except KeyError:
                        watt_trans = linkInfo[(src,dst)]["WATT_TRANS"]
                    energy_trans = time_link * watt_trans

                    try:
                        watt_recv = linkInfo[(src,dst)][f"WATT_RECV_{src}-{dst}"]
                    except KeyError:
                        watt_recv = linkInfo[(src,dst)]["WATT_RECV"]
                    energy_recv = time_link * watt_recv

                if tail_enabled:
                    energy_trans += float(df_tail_grouped.loc[link].tail_tx_wh)
                    energy_recv += float(df_tail_grouped.loc[link].tail_rx_wh)

                if src not in results:
                    results[src] = {"model": nodeInfo[src]["model"], "type": nodeInfo[src]["type"],
                                    "watt_trans":0.0, "watt_recv":0.0}
                if dst not in results:
                    results[dst] = {"model": nodeInfo[dst]["model"], "type": nodeInfo[dst]["type"],
                                    "watt_trans":0.0, "watt_recv":0.0}
                results[src]["watt_trans"] += energy_trans
                results[dst]["watt_recv"] += energy_recv

            if self.df_offloading is not None:
                needed = {
                    "current_mobile_topology_id",
                    "current_fog_topology_id",
                }
                if needed.issubset(set(self.df_offloading.columns)):
                    df_off = self.df_offloading.copy()
                    if "sim_time_s" in df_off.columns:
                        df_off = df_off.sort_values("sim_time_s")

                    if {"migration_edge_energy_wh", "migration_fog_energy_wh"}.issubset(df_off.columns):
                        edge_wh_series = df_off["migration_edge_energy_wh"].fillna(0.0).astype(float)
                        fog_wh_series = df_off["migration_fog_energy_wh"].fillna(0.0).astype(float)
                    else:
                        edge_wh_series = pd.Series(0.0, index=df_off.index)
                        fog_wh_series = pd.Series(0.0, index=df_off.index)

                    for idx, row in df_off.iterrows():
                        edge_id = row.get("current_mobile_topology_id", np.nan)
                        fog_id = row.get("current_fog_topology_id", np.nan)
                        edge_wh = float(edge_wh_series.loc[idx])
                        fog_wh = float(fog_wh_series.loc[idx])

                        if pd.isna(edge_id) or pd.isna(fog_id):
                            continue

                        edge_id = int(edge_id)
                        fog_id = int(fog_id)
                        if edge_id not in nodeInfo or fog_id not in nodeInfo:
                            continue

                        if edge_id not in results:
                            results[edge_id] = {"model": nodeInfo[edge_id]["model"], "type": nodeInfo[edge_id]["type"],
                                                "watt_trans":0.0, "watt_recv":0.0}
                        if fog_id not in results:
                            results[fog_id] = {"model": nodeInfo[fog_id]["model"], "type": nodeInfo[fog_id]["type"],
                                               "watt_trans":0.0, "watt_recv":0.0}

                        if not pd.isna(edge_wh):
                            results[edge_id]["watt_trans"] += float(edge_wh)
                        if not pd.isna(fog_wh):
                            results[fog_id]["watt_recv"] += float(fog_wh)
        else:
            for node_key in nodeInfo:
                if not nodeInfo[node_key]["uptime"][1]:
                    end = totaltime
                start = nodeInfo[node_key]["uptime"][0]
                uptime = end-start
                results[node_key] = {"model":nodeInfo[node_key]["model"],
                                     "type":nodeInfo[node_key]["type"],
                                     "watt":uptime*nodeInfo[node_key]["WATT"],
                                     "uptime":uptime
                                     }

        return results

    # def get_cost_cloud(self, topology):
    #     cost = 0.0
    #     nodeInfo = topology.get_info()
    #     results = {}
    #     # Tiempo de actividad / runeo
    #     if "time_response" not in self.df.columns:  # cached
    #         self.__compute_times_df()
    #
    #     nodes = self.df.groupby("TOPO.dst").agg({"time_service": "sum"})
    #
    #     for id_node in nodes.index:
    #         if nodeInfo[id_node]["type"] == Entity.ENTITY_CLOUD:
    #             results[id_node] = {"model": nodeInfo[id_node]["model"], "type": nodeInfo[id_node]["type"],
    #                                 "watt": nodes.loc[id_node].time_service * nodeInfo[id_node]["WATT"]}
    #             cost += nodes.loc[id_node].time_service * nodeInfo[id_node]["COST"]
    #     return cost,results

    def showLoops(self,time_loops):
        results = self.average_loop_response(time_loops)
        for i, loop in enumerate(time_loops):
            print ("\t\t%i - %s :\t %f" % (i, str(loop), results[i]))
        return results



    # multiplier - good for periodic processes to scale results for a longer run
    def showResults(self, total_time, topology, time_loops=None, multiplier=1):
        if multiplier <= 0:
            multiplier = 1
        elif multiplier > 1:
            print ("\tNote: Results are multiplied by %f" % multiplier)

        print ("\tSimulation Time: %0.2f" % (total_time * multiplier))

        if time_loops is not None:
            print ("\tApplication loops delays:")
            results = self.average_loop_response(time_loops)
            for i, loop in enumerate(time_loops):
                print ("\t\t%i - %s :\t %f" % (i, str(loop), results[i] * multiplier))

        print ("\tEnergy Consumed (Wh by UpTime):")
        values = dict(sorted(self.get_watt(total_time, topology, Metrics.WATT_UPTIME).items()))
        for k, node in values.items():
            print ("\t\t%i - %s :\t %.6f" % (k, node["model"], node["watt"] * multiplier))

        print ("\tEnergy Consumed by Service (Wh by Service Time):")
        values = dict(sorted(self.get_watt(total_time, topology, Metrics.WATT_SERVICE).items()))
        for k, node in values.items():
            print ("\t\t%i - %s :\t %.6f" % (k, node["model"], node["watt"] * multiplier))

        print ("\tEnergy Consumed by Transmission Link (Wh by Link Time):")
        values = dict(sorted(self.get_watt(total_time, topology, Metrics.WATT_LINK).items()))
        for k, node in values.items():
            print ("\t\t%i - %s :\t Transmit: %.6f  Recv: %.6f" % (k, node["model"],
                                                        node["watt_trans"] * multiplier,
                                                        node["watt_recv"] * multiplier))

        # print ("\tCost of execution in cloud:")
        # total, values = self.get_cost_cloud(topology)
        # print ("\t\t%.8f" % total)

        print ("\tNetwork bytes transmitted:")
        bytes = self.bytes_transmitted() * multiplier
        print ("\t\t%.1f (%.2f MiB)" % (bytes, (bytes / (1024.0 * 1024.0))))


    def showResults2(self, total_time, time_loops=None):
        print ("\tSimulation Time: %0.2f" % total_time)

        if time_loops is not None:
            print ("\tApplication loops delays:")
            results = self.average_loop_response(time_loops)
            for i, loop in enumerate(time_loops):
                print ("\t\t%i - %s :\t %f" % (i, str(loop), results[i]))

        print ("\tNetwork bytes transmitted:")
        print ("\t\t%.1f" % self.bytes_transmitted())

    """ONLYE THE FIRST ONE : DEBUG"""
    def valueLoop(self, total_time, time_loops=None):
            if time_loops is not None:
                results = self.average_loop_response(time_loops)
                for i, loop in enumerate(time_loops):
                    return results[i]

    def average_messages_not_transmitted(self):
        return np.mean(self.df_link.buffer)

    def peak_messages_not_transmitted(self):
        return np.max(self.df_link.buffer)

    def messages_not_transmitted(self):
        return self.df_link.buffer[-1:]

    def get_df_modules(self):
        g = self.df.groupby(["module", "DES.dst"]).agg({"service": ['mean', 'sum', 'count']})
        return g.reset_index()

    def get_df_service_utilization(self,service,time):
        """
        Returns the utilization(%) of a specific module
        """
        g = self.df.groupby(["module", "DES.dst"]).agg({"service": ['mean', 'sum', 'count']})
        g.reset_index(inplace=True)
        h = pd.DataFrame()
        h["module"] = g[g.module == service].module
        h["utilization"] = g[g.module == service]["service"]["sum"]*100 / time
        return h

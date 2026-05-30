from pathlib import Path

import pandas as pd

import config as cfg
from mobility import MobilityModel, MobilityPlacement


class LPOptimizationPlacement(MobilityPlacement):
    """
    Placement policy that combines:
    1) Fog module mobility placement (nearest fog node),
    2) LP EDGE/FOG mode decision and decision logging.
    """

    def __init__(
        self,
        mobility_model: MobilityModel,
        app_name: str,
        output_path: Path,
        mode_getter,
        mode_setter,
        rssi_from_distance_dbm,
        wifi_throughput_mbps_from_rssi,
        sensor_wifi_throughput_mbps_from_rssi,
        tcp_retransmission_rate_from_rssi,
        sensor_ble_energy_wh_per_mb,
        sensor_ble_data_bw_mb_s,
        sensor_ble_rx_power_w,
        sensor_ble_tail_energy_wh_per_transfer,
        sensor_wifi_data_powers_w_from_rssi,
        sensor_wifi_promotion_energy_wh_per_transfer,
        sensor_wifi_tail_energy_wh_per_transfer,
        mobile_to_fog_wifi_powers_w_from_rssi,
        fog_to_mobile_wifi_powers_w_from_rssi,
        tail_energy_wh_per_transfer,
        kb_to_mb,
        mi_to_instructions,
        solve_lp_two_mode,
        ips_to_ipt,
        watt_to_wpt,
        **kwargs,
    ):
        super().__init__(mobility_model=mobility_model, app_name=app_name, **kwargs)
        self.output_path = output_path
        self.mode_getter = mode_getter
        self.mode_setter = mode_setter

        self.rssi_from_distance_dbm = rssi_from_distance_dbm
        self.wifi_throughput_mbps_from_rssi = wifi_throughput_mbps_from_rssi
        self.sensor_wifi_throughput_mbps_from_rssi = sensor_wifi_throughput_mbps_from_rssi
        self.tcp_retransmission_rate_from_rssi = tcp_retransmission_rate_from_rssi
        self.sensor_ble_energy_wh_per_mb = sensor_ble_energy_wh_per_mb
        self.sensor_ble_data_bw_mb_s = sensor_ble_data_bw_mb_s
        self.sensor_ble_rx_power_w = sensor_ble_rx_power_w
        self.sensor_ble_tail_energy_wh_per_transfer = sensor_ble_tail_energy_wh_per_transfer
        self.sensor_wifi_data_powers_w_from_rssi = sensor_wifi_data_powers_w_from_rssi
        self.sensor_wifi_promotion_energy_wh_per_transfer = (
            sensor_wifi_promotion_energy_wh_per_transfer
        )
        self.sensor_wifi_tail_energy_wh_per_transfer = (
            sensor_wifi_tail_energy_wh_per_transfer
        )
        self.mobile_to_fog_wifi_powers_w_from_rssi = mobile_to_fog_wifi_powers_w_from_rssi
        self.fog_to_mobile_wifi_powers_w_from_rssi = fog_to_mobile_wifi_powers_w_from_rssi
        self.tail_energy_wh_per_transfer = tail_energy_wh_per_transfer
        self.kb_to_mb = kb_to_mb
        self.mi_to_instructions = mi_to_instructions
        self.solve_lp_two_mode = solve_lp_two_mode
        self.ips_to_ipt = ips_to_ipt
        self.watt_to_wpt = watt_to_wpt

        self.buffer = []
        self.header_written = False
        self.last_mode = str(self.mode_getter()).upper()

    def run(self, sim):
        # Keep Fog service attached to nearest fog node.
        super().run(sim)

        step = int(sim.env.now)
        if step % max(int(cfg.OFFLOADING_DECISION_PERIOD_S), 1) != 0:
            return

        distances = self.mobility_model.distances_for_step(step)
        if not distances:
            return

        nearest_fog, nearest_distance_m = min(distances, key=lambda item: item[1])
        fog_des = sim.alloc_module.get(self.app_name, {}).get("Fog", [])
        current_fog_topology_id = sim.alloc_DES.get(fog_des[0], None) if fog_des else None
        mobile_des = sim.alloc_module.get(self.app_name, {}).get("Mobile", [])
        current_mobile_topology_id = sim.alloc_DES.get(mobile_des[0], None) if mobile_des else None
        rssi_fog_dbm = self.rssi_from_distance_dbm(nearest_distance_m)
        wifi_bw_mb_s = self.wifi_throughput_mbps_from_rssi(rssi_fog_dbm)
        sensor_wifi_bw_mb_s = self.sensor_wifi_throughput_mbps_from_rssi(rssi_fog_dbm)
        tcp_retx_rate = self.tcp_retransmission_rate_from_rssi(rssi_fog_dbm)
        attempts_factor = 1.0 / max(1.0 - tcp_retx_rate, 1e-12)

        task_input_mb = self.kb_to_mb(cfg.TASK1_DATA_SIZE_KB)
        task_result_mb = 740.0 / (1024.0 * 1024.0)
        migration_mb = self.kb_to_mb(cfg.TASK1_MIGRATION_STATE_SIZE_KB)

        ble_bw_mb_s = self.sensor_ble_data_bw_mb_s()
        delay_sensor_edge = task_input_mb / max(ble_bw_mb_s, 1e-12)
        delay_edge_fog = task_result_mb / max(wifi_bw_mb_s, 1e-12) * attempts_factor
        delay_sensor_fog = task_input_mb / max(sensor_wifi_bw_mb_s, 1e-12) * attempts_factor

        mobile_ipt = self.ips_to_ipt(1.9 * 10**9)
        fog_ipt = self.ips_to_ipt(500 * 10**6)
        proc_instructions = self.mi_to_instructions(cfg.TASK1_COMPLEXITY_MI)
        proc_delay_edge = proc_instructions / max(mobile_ipt, 1e-12)
        proc_delay_fog = proc_instructions / max(fog_ipt, 1e-12)

        latency_edge = delay_sensor_edge + proc_delay_edge + delay_edge_fog
        latency_fog = delay_sensor_fog + proc_delay_fog

        ble_transfer_tu = task_input_mb / max(ble_bw_mb_s, 1e-12)
        _, sensor_ble_data_wh_per_mb, sensor_ble_tail_wh = self.sensor_ble_energy_wh_per_mb()
        sensor_ble_tx_wh = sensor_ble_data_wh_per_mb * task_input_mb + sensor_ble_tail_wh

        edge_ble_rx_wh = self.watt_to_wpt(self.sensor_ble_rx_power_w()) * ble_transfer_tu
        edge_ble_rx_wh += self.sensor_ble_tail_energy_wh_per_transfer()
        edge_proc_wh = self.watt_to_wpt(1.3) * proc_delay_edge
        edge_wifi_tx_w, fog_wifi_rx_w = self.mobile_to_fog_wifi_powers_w_from_rssi(rssi_fog_dbm)
        fog_wifi_tx_w, edge_wifi_rx_w = self.fog_to_mobile_wifi_powers_w_from_rssi(rssi_fog_dbm)
        edge_tx_to_fog_wh = self.watt_to_wpt(edge_wifi_tx_w) * (task_result_mb / max(wifi_bw_mb_s, 1e-12) * attempts_factor)
        edge_tx_to_fog_wh += self.tail_energy_wh_per_transfer()
        energy_edge = sensor_ble_tx_wh + edge_ble_rx_wh + edge_proc_wh + edge_tx_to_fog_wh

        sensor_wifi_tx_w, _, sensor_wifi_available = self.sensor_wifi_data_powers_w_from_rssi(rssi_fog_dbm)
        if sensor_wifi_available:
            sensor_tx_to_fog_wh = self.watt_to_wpt(sensor_wifi_tx_w) * (task_input_mb / max(sensor_wifi_bw_mb_s, 1e-12) * attempts_factor)
            sensor_tx_to_fog_wh += self.sensor_wifi_promotion_energy_wh_per_transfer()
            sensor_tx_to_fog_wh += self.sensor_wifi_tail_energy_wh_per_transfer()
        else:
            sensor_tx_to_fog_wh = 0.0

        edge_rx_from_fog_wh = self.watt_to_wpt(edge_wifi_rx_w) * (task_result_mb / max(wifi_bw_mb_s, 1e-12) * attempts_factor)
        edge_rx_from_fog_wh += self.tail_energy_wh_per_transfer()
        energy_fog = sensor_tx_to_fog_wh + edge_rx_from_fog_wh

        c_edge = cfg.ALPHA_LATENCY * latency_edge + cfg.BETA_ENERGY * energy_edge
        c_fog = cfg.ALPHA_LATENCY * latency_fog + cfg.BETA_ENERGY * energy_fog
        if not sensor_wifi_available:
            chosen_mode = "EDGE"
        else:
            chosen_mode = self.solve_lp_two_mode(c_edge, c_fog)

        migration_edge_energy_wh = 0.0
        migration_fog_energy_wh = 0.0
        if chosen_mode != self.last_mode and chosen_mode == "FOG":
            migration_time_tu = migration_mb / max(wifi_bw_mb_s, 1e-12) * attempts_factor
            migration_edge_energy_wh = self.watt_to_wpt(edge_wifi_tx_w) * migration_time_tu
            migration_edge_energy_wh += self.tail_energy_wh_per_transfer()
            migration_fog_energy_wh = self.watt_to_wpt(fog_wifi_rx_w) * migration_time_tu
        migration_energy_wh = migration_edge_energy_wh + migration_fog_energy_wh

        real_energy_edge = 0.0
        real_energy_sensors = 0.0
        if self.last_mode == "EDGE":
            real_energy_edge += edge_ble_rx_wh + edge_proc_wh + edge_tx_to_fog_wh
            real_energy_sensors += sensor_ble_tx_wh
            real_energy_edge += migration_edge_energy_wh
        else:
            real_energy_edge += edge_rx_from_fog_wh
            real_energy_sensors += sensor_tx_to_fog_wh

        self.mode_setter(chosen_mode)
        self.last_mode = chosen_mode

        self.buffer.append(
            {
                "sim_time_s": step,
                "nearest_fog_topology_id": nearest_fog.topo_id,
                "current_fog_topology_id": current_fog_topology_id,
                "current_mobile_topology_id": current_mobile_topology_id,
                "nearest_fog_distance_m": nearest_distance_m,
                "nearest_fog_rssi_dbm": rssi_fog_dbm,
                "wifi_bw_mb_s": wifi_bw_mb_s,
                "tcp_retransmission_rate": tcp_retx_rate,
                "latency_edge_tu": latency_edge,
                "latency_fog_tu": latency_fog,
                "energy_edge_wh": energy_edge,
                "energy_fog_wh": energy_fog,
                "objective_edge": c_edge,
                "objective_fog": c_fog,
                "chosen_mode": chosen_mode,
                "migration_edge_energy_wh": migration_edge_energy_wh,
                "migration_fog_energy_wh": migration_fog_energy_wh,
                "migration_energy_wh": migration_energy_wh,
                "real_energy_edge_wh": real_energy_edge,
                "real_energy_sensors_wh": real_energy_sensors,
            }
        )
        self.flush()

    def flush(self):
        if not self.buffer:
            return
        df = pd.DataFrame(self.buffer)
        mode = "a" if self.header_written else "w"
        df.to_csv(self.output_path, mode=mode, header=(not self.header_written), index=False)
        self.header_written = True
        self.buffer.clear()

from .ack_bitrate_estimator import AckBitrateEstimator
from .trendline_estimator import TrendlineEstimator
from .packet_info import PacketInfo
from .packet_record import PacketRecord
from .inter_arrival import InterArrival
import logging
# logging.basicConfig(level=logging.INFO)


class Estimator:
    '''
    Estimator 用于从获得的数据包信息里面获取接收端的接受速率，丢包率以及Delay jitter
    由于不需要进行实际的速率调控，这里省去了速率调控相关的逻辑，只保留了速率估计的逻辑
    '''

    def __init__(self) -> None:
        self.ack_bitrate_estimator = AckBitrateEstimator()
        self.trendline_estimator = TrendlineEstimator(threshold_gain=10)
        self.packet_record = PacketRecord()
        self.inter_arrival = InterArrival()

        self.time_now = 0
        self.first_time = 0

    def parse_packets(self, packets):
        '''
        从数据包中解析出速率和延迟抖动，返回速率，丢包和Trendline
        '''
        # 调用ack_bitrate的bitrate计算逻辑
        self.ack_bitrate_estimator.ack_estimator_incoming(packets)
        receiving_rate = self.ack_bitrate_estimator.ack_estimator_bitrate_bps()
        # 调用PacketRecord的计算逻辑
        for pkt in packets:
            pki = PacketInfo()
            pki.payload_type = pkt['payload_type']
            pki.ssrc = pkt['ssrc']
            pki.sequence_number = pkt['sequence_number']
            pki.send_timestamp = pkt['send_time_ms']
            pki.receive_timestamp = pkt['arrival_time_ms']
            pki.padding_length = pkt['padding_length']
            pki.header_length = pkt['header_length']
            pki.payload_size = pkt['payload_size']
            self.time_now = pkt["time_now"]
            self.packet_record.on_receive(pki)

        receiving_rate_record = self.packet_record.calculate_receiving_rate()
        # print(f"receiving_rate_record: {receiving_rate_record}, receiving_rate: {receiving_rate}")
        loss_ratio = self.packet_record.calculate_loss_ratio()
        delay = self.packet_record.calculate_average_delay()
        delay = delay if delay > 0 else 50
        self.packet_record.clear()
        # 调用trendline的计算逻辑
        for pkt in packets:
            if pkt["send_time_ms"] < self.first_time:
                continue
            self.__delay_bwe_progress(pkt, self.time_now)
        trendline = self.trendline_estimator.trendline_slope() * \
            min(self.trendline_estimator.num_of_deltas, 60)

        # logging.info(f"now_ts={self.time_now}, "
        #              f"receiving_rate_record={receiving_rate_record}, "
        #              f"receiving_rate={receiving_rate}, "
        #              f"loss_ratio={loss_ratio}, "
        #              f"trendline={trendline}")
        return receiving_rate, loss_ratio, trendline, delay

    def __delay_bwe_progress(self, pkt, now_ts):
        # 从数据包中解析出延迟抖动
        ts_delta = 0
        t_delta = 0
        size_delta = 0

        self.last_seen_ms = now_ts
        packet_arrival_ts = pkt["arrival_time_ms"]
        packet_payload_size = pkt["payload_size"]

        timestamp = pkt["send_time_ms"] - self.first_time
        ret, ts_delta, t_delta, size_delta = self.inter_arrival.inter_arrival_compute_deltas(
            timestamp, packet_arrival_ts, now_ts, packet_payload_size, ts_delta, t_delta, size_delta)
        if ret == 0:
            self.trendline_estimator.trendline_update(
                t_delta, ts_delta, packet_arrival_ts)

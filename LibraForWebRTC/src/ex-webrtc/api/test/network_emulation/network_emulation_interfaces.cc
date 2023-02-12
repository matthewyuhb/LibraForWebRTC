/*
 *  Copyright (c) 2019 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#include "api/test/network_emulation/network_emulation_interfaces.h"

#include "rtc_base/net_helper.h"

namespace webrtc {
EmulatedIpPacket::EmulatedIpPacket(const rtc::SocketAddress& from,
                                   const rtc::SocketAddress& to,
                                   rtc::CopyOnWriteBuffer data,
                                   Timestamp arrival_time,
                                   uint16_t application_overhead)
    : from(from),
      to(to),
      data(data),
      headers_size(to.ipaddr().overhead() + application_overhead +
                   cricket::kUdpHeaderSize),
      arrival_time(arrival_time) {
  // GYM would not send real traffic
  // so the payload may not belong a valid family
#if !defined(GYM)
  RTC_DCHECK(to.family() == AF_INET || to.family() == AF_INET6);
#endif
}

}  // namespace webrtc

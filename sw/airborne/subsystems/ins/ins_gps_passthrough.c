/*
 * Copyright (C) 2004-2012 The Paparazzi Team
 *
 * This file is part of paparazzi.
 *
 * paparazzi is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2, or (at your option)
 * any later version.
 *
 * paparazzi is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with paparazzi; see the file COPYING.  If not, write to
 * the Free Software Foundation, 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

/**
 * @file subsystems/ins/ins_gps_passthrough.c
 *
 * Simply converts GPS ECEF position and velocity to NED
 * and passes it through to the state interface.
 */

#include "subsystems/ins.h"

#include <inttypes.h>
#include <math.h>

#include "state.h"
#include "subsystems/gps.h"

#ifndef USE_INS_NAV_INIT
#define USE_INS_NAV_INIT TRUE
PRINT_CONFIG_MSG("USE_INS_NAV_INIT defaulting to TRUE")
#endif

#if USE_INS_NAV_INIT
#include "generated/flight_plan.h"
#endif

struct InsGpsPassthrough {
  struct LtpDef_i  ltp_def;
  bool_t           ltp_initialized;

  /* output LTP NED */
  struct NedCoor_i ltp_pos;
  struct NedCoor_i ltp_speed;
  struct NedCoor_i ltp_accel;
};

struct InsGpsPassthrough ins_impl;

#if PERIODIC_TELEMETRY
#include "subsystems/datalink/telemetry.h"

static void send_ins(void) {
  DOWNLINK_SEND_INS(DefaultChannel, DefaultDevice,
      &ins_impl.ltp_pos.x, &ins_impl.ltp_pos.y, &ins_impl.ltp_pos.z,
      &ins_impl.ltp_speed.x, &ins_impl.ltp_speed.y, &ins_impl.ltp_speed.z,
      &ins_impl.ltp_accel.x, &ins_impl.ltp_accel.y, &ins_impl.ltp_accel.z);
}

static void send_ins_z(void) {
  static const float fake_baro_z = 0.0;
  DOWNLINK_SEND_INS_Z(DefaultChannel, DefaultDevice,
      &fake_baro_z, &ins_impl.ltp_pos.z, &ins_impl.ltp_speed.z, &ins_impl.ltp_accel.z);
}

static void send_ins_ref(void) {
  static const float fake_qfe = 0.0;
  if (ins_impl.ltp_initialized) {
    DOWNLINK_SEND_INS_REF(DefaultChannel, DefaultDevice,
        &ins_impl.ltp_def.ecef.x, &ins_impl.ltp_def.ecef.y, &ins_impl.ltp_def.ecef.z,
        &ins_impl.ltp_def.lla.lat, &ins_impl.ltp_def.lla.lon, &ins_impl.ltp_def.lla.alt,
        &ins_impl.ltp_def.hmsl, &fake_qfe);
  }
}
#endif

void ins_init(void) {

#if USE_INS_NAV_INIT
  struct LlaCoor_i llh_nav0; /* Height above the ellipsoid */
  llh_nav0.lat = NAV_LAT0;
  llh_nav0.lon = NAV_LON0;
  /* NAV_ALT0 = ground alt above msl, NAV_MSL0 = geoid-height (msl) over ellipsoid */
  llh_nav0.alt = NAV_ALT0 + NAV_MSL0;

  struct EcefCoor_i ecef_nav0;
  ecef_of_lla_i(&ecef_nav0, &llh_nav0);

  ltp_def_from_ecef_i(&ins_impl.ltp_def, &ecef_nav0);
  ins_impl.ltp_def.hmsl = NAV_ALT0;
  stateSetLocalOrigin_i(&ins_impl.ltp_def);

  ins_impl.ltp_initialized = TRUE;
#else
  ins_impl.ltp_initialized  = FALSE;
#endif

  INT32_VECT3_ZERO(ins_impl.ltp_pos);
  INT32_VECT3_ZERO(ins_impl.ltp_speed);
  INT32_VECT3_ZERO(ins_impl.ltp_accel);

#if PERIODIC_TELEMETRY
  register_periodic_telemetry(DefaultPeriodic, "INS", send_ins);
  register_periodic_telemetry(DefaultPeriodic, "INS_Z", send_ins_z);
  register_periodic_telemetry(DefaultPeriodic, "INS_REF", send_ins_ref);
#endif
}

void ins_periodic(void) {
  if (ins_impl.ltp_initialized)
    ins.status = INS_RUNNING;
}


void ins_reset_local_origin(void) {
  ltp_def_from_ecef_i(&ins_impl.ltp_def, &gps.ecef_pos);
  ins_impl.ltp_def.lla.alt = gps.lla_pos.alt;
  ins_impl.ltp_def.hmsl = gps.hmsl;
  stateSetLocalOrigin_i(&ins_impl.ltp_def);
  ins_impl.ltp_initialized = TRUE;
}

void ins_reset_altitude_ref(void) {
  struct LlaCoor_i lla = {
    state.ned_origin_i.lla.lon,
    state.ned_origin_i.lla.lat,
    gps.lla_pos.alt
  };
  ltp_def_from_lla_i(&ins_impl.ltp_def, &lla),
  ins_impl.ltp_def.hmsl = gps.hmsl;
  stateSetLocalOrigin_i(&ins_impl.ltp_def);
}

void ins_update_gps(void) {
  if (gps.fix == GPS_FIX_3D) {
    if (!ins_impl.ltp_initialized) {
      ins_reset_local_origin();
    }

    /* simply scale and copy pos/speed from gps */
    struct NedCoor_i gps_pos_cm_ned;
    ned_of_ecef_point_i(&gps_pos_cm_ned, &ins_impl.ltp_def, &gps.ecef_pos);
    INT32_VECT3_SCALE_2(ins_impl.ltp_pos, gps_pos_cm_ned,
                        INT32_POS_OF_CM_NUM, INT32_POS_OF_CM_DEN);
    stateSetPositionNed_i(&ins_impl.ltp_pos);

    struct NedCoor_i gps_speed_cm_s_ned;
    ned_of_ecef_vect_i(&gps_speed_cm_s_ned, &ins_impl.ltp_def, &gps.ecef_vel);
    INT32_VECT3_SCALE_2(ins_impl.ltp_speed, gps_speed_cm_s_ned,
                        INT32_SPEED_OF_CM_S_NUM, INT32_SPEED_OF_CM_S_DEN);
    stateSetSpeedNed_i(&ins_impl.ltp_speed);
  }
}

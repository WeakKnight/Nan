"""
Sun position calculation module based on NOAA solar position algorithm.

Computes accurate sun position (elevation, azimuth) and related astronomical data
(sunrise, sunset, solar noon) for any location and date/time on Earth.

Reference: NOAA Solar Calculator (https://www.esrl.noaa.gov/gmd/grad/solcalc/)
Implementation based on UE SunPosition plugin.
"""

import math
from dataclasses import dataclass
from datetime import datetime, time


@dataclass
class SunPositionData:
    """
    Data class containing sun position calculation results.
    
    Attributes:
        elevation: Sun elevation angle in degrees (0 = horizon, 90 = zenith)
        corrected_elevation: Elevation corrected for atmospheric refraction
        azimuth: Sun azimuth angle in degrees (0 = North, clockwise)
        sun_direction: Unit direction vector in Y-up coordinate system (x, y, z)
        sunrise_time: Local sunrise time
        sunset_time: Local sunset time
        solar_noon: Local solar noon time
    """
    elevation: float
    corrected_elevation: float
    azimuth: float
    sun_direction: tuple[float, float, float]
    sunrise_time: time
    sunset_time: time
    solar_noon: time


class SunPosition:
    """
    Calculator for sun position based on geographic location and date/time.
    
    Uses the NOAA solar position algorithm based on Julian Day calculations
    to compute accurate sun positions for any location on Earth.
    """
    
    @staticmethod
    def calculate(
        latitude: float,
        longitude: float,
        year: int,
        month: int,
        day: int,
        hour: int,
        minute: int,
        second: int,
        timezone: float = 0.0,
        daylight_saving: bool = False
    ) -> SunPositionData:
        """
        Calculate sun position for given location and date/time.
        
        Args:
            latitude: Geographic latitude in degrees (-90 to 90, positive = North)
            longitude: Geographic longitude in degrees (-180 to 180, positive = East)
            year: Year (e.g., 2024)
            month: Month (1-12)
            day: Day of month (1-31)
            hour: Hour (0-23)
            minute: Minute (0-59)
            second: Second (0-59)
            timezone: Timezone offset from UTC in hours (e.g., 8.0 for UTC+8)
            daylight_saving: Whether daylight saving time is in effect
            
        Returns:
            SunPositionData containing elevation, azimuth, direction vector and times
        """
        # Validate date
        try:
            datetime(year, month, day, hour, minute, second)
        except ValueError as e:
            raise ValueError(f"Invalid date/time: {e}")
        
        # Apply timezone and daylight saving offset
        time_offset = timezone
        if daylight_saving:
            time_offset += 1.0
        
        # Convert latitude to radians
        latitude_rad = math.radians(latitude)
        
        # Get Julian Day (number of days since Jan 1, 4713 BC)
        julian_day = SunPosition._get_julian_day(year, month, day, hour, minute, second)
        julian_century = (julian_day - 2451545.0) / 36525.0
        
        # Sun's mean longitude (degrees), referred to mean equinox of julian date
        geom_mean_long_sun_deg = math.fmod(
            280.46646 + julian_century * (36000.76983 + julian_century * 0.0003032), 
            360.0
        )
        
        # Sun's mean anomaly (degrees)
        geom_mean_anom_sun_deg = 357.52911 + julian_century * (35999.05029 - 0.0001537 * julian_century)
        geom_mean_anom_sun_rad = math.radians(geom_mean_anom_sun_deg)
        
        # Earth's orbit eccentricity
        eccent_earth_orbit = 0.016708634 - julian_century * (0.000042037 + 0.0000001267 * julian_century)
        
        # Sun's equation of center
        sun_eq_of_ctr = (
            math.sin(geom_mean_anom_sun_rad) * (1.914602 - julian_century * (0.004817 + 0.000014 * julian_century))
            + math.sin(2.0 * geom_mean_anom_sun_rad) * (0.019993 - 0.000101 * julian_century)
            + math.sin(3.0 * geom_mean_anom_sun_rad) * 0.000289
        )
        
        # Sun's true longitude (degrees)
        sun_true_long_deg = geom_mean_long_sun_deg + sun_eq_of_ctr
        
        # Sun's apparent longitude (degrees)
        sun_app_long_deg = sun_true_long_deg - 0.00569 - 0.00478 * math.sin(
            math.radians(125.04 - 1934.136 * julian_century)
        )
        sun_app_long_rad = math.radians(sun_app_long_deg)
        
        # Earth's mean obliquity of the ecliptic (degrees)
        mean_obliq_ecliptic_deg = (
            23.0 + (26.0 + (21.448 - julian_century * (46.815 + julian_century * (0.00059 - julian_century * 0.001813))) / 60.0) / 60.0
        )
        
        # Oblique correction (degrees)
        obliq_corr_deg = mean_obliq_ecliptic_deg + 0.00256 * math.cos(
            math.radians(125.04 - 1934.136 * julian_century)
        )
        obliq_corr_rad = math.radians(obliq_corr_deg)
        
        # Sun's declination (radians)
        sun_declin_rad = math.asin(math.sin(obliq_corr_rad) * math.sin(sun_app_long_rad))
        
        # Variable Y for equation of time
        var_y = math.pow(math.tan(obliq_corr_rad / 2.0), 2.0)
        geom_mean_long_sun_rad = math.radians(geom_mean_long_sun_deg)
        
        # Equation of time (minutes)
        eq_of_time_minutes = 4.0 * math.degrees(
            var_y * math.sin(2.0 * geom_mean_long_sun_rad)
            - 2.0 * eccent_earth_orbit * math.sin(geom_mean_anom_sun_rad)
            + 4.0 * eccent_earth_orbit * var_y * math.sin(geom_mean_anom_sun_rad) * math.cos(2.0 * geom_mean_long_sun_rad)
            - 0.5 * var_y * var_y * math.sin(4.0 * geom_mean_long_sun_rad)
            - 1.25 * eccent_earth_orbit * eccent_earth_orbit * math.sin(2.0 * geom_mean_anom_sun_rad)
        )
        
        # Hour angle of sunrise (degrees)
        cos_ha_sunrise = (
            math.cos(math.radians(90.833)) / (math.cos(latitude_rad) * math.cos(sun_declin_rad))
            - math.tan(latitude_rad) * math.tan(sun_declin_rad)
        )
        
        # Clamp to valid range for acos
        cos_ha_sunrise = max(-1.0, min(1.0, cos_ha_sunrise))
        ha_sunrise_deg = math.degrees(math.acos(cos_ha_sunrise))
        
        # Solar noon, sunrise and sunset times (as fraction of day in local standard time)
        solar_noon_lst = (720.0 - 4.0 * longitude - eq_of_time_minutes + time_offset * 60.0) / 1440.0
        sunrise_time_lst = solar_noon_lst - ha_sunrise_deg * 4.0 / 1440.0
        sunset_time_lst = solar_noon_lst + ha_sunrise_deg * 4.0 / 1440.0
        
        # True solar time (minutes)
        time_of_day_minutes = hour * 60.0 + minute + second / 60.0
        true_solar_time_minutes = math.fmod(
            time_of_day_minutes + eq_of_time_minutes + 4.0 * longitude - 60.0 * time_offset,
            1440.0
        )
        
        # Hour angle of current time (degrees)
        if true_solar_time_minutes < 0:
            hour_angle_deg = true_solar_time_minutes / 4.0 + 180.0
        else:
            hour_angle_deg = true_solar_time_minutes / 4.0 - 180.0
        hour_angle_rad = math.radians(hour_angle_deg)
        
        # Solar zenith angle
        solar_zenith_angle_rad = math.acos(
            math.sin(latitude_rad) * math.sin(sun_declin_rad)
            + math.cos(latitude_rad) * math.cos(sun_declin_rad) * math.cos(hour_angle_rad)
        )
        solar_zenith_angle_deg = math.degrees(solar_zenith_angle_rad)
        
        # Solar elevation angle
        solar_elevation_angle_deg = 90.0 - solar_zenith_angle_deg
        solar_elevation_angle_rad = math.radians(solar_elevation_angle_deg)
        tan_of_solar_elevation = math.tan(solar_elevation_angle_rad)
        
        # Atmospheric refraction correction (degrees)
        approx_atmospheric_refraction_deg = 0.0
        if solar_elevation_angle_deg <= 85.0:
            if solar_elevation_angle_deg > 5.0:
                approx_atmospheric_refraction_deg = (
                    58.1 / tan_of_solar_elevation
                    - 0.07 / math.pow(tan_of_solar_elevation, 3)
                    + 0.000086 / math.pow(tan_of_solar_elevation, 5)
                ) / 3600.0
            elif solar_elevation_angle_deg > -0.575:
                approx_atmospheric_refraction_deg = (
                    1735.0 + solar_elevation_angle_deg * (
                        -518.2 + solar_elevation_angle_deg * (
                            103.4 + solar_elevation_angle_deg * (
                                -12.79 + solar_elevation_angle_deg * 0.711
                            )
                        )
                    )
                ) / 3600.0
            else:
                approx_atmospheric_refraction_deg = -20.772 / tan_of_solar_elevation / 3600.0
        
        # Corrected solar elevation
        solar_elevation_corrected_deg = solar_elevation_angle_deg + approx_atmospheric_refraction_deg
        
        # Solar azimuth angle (degrees, clockwise from North)
        tmp = math.degrees(math.acos(
            (math.sin(latitude_rad) * math.cos(solar_zenith_angle_rad) - math.sin(sun_declin_rad))
            / (math.cos(latitude_rad) * math.sin(solar_zenith_angle_rad))
        ))
        
        if hour_angle_deg > 0.0:
            solar_azimuth_angle_deg = math.fmod(tmp + 180.0, 360.0)
        else:
            solar_azimuth_angle_deg = math.fmod(540.0 - tmp, 360.0)
        
        # Convert to direction vector (Y-up coordinate system)
        sun_direction = SunPosition.elevation_azimuth_to_direction(
            solar_elevation_angle_deg, 
            solar_azimuth_angle_deg
        )
        
        # Convert times from day fraction to time objects
        sunrise_time = SunPosition._day_fraction_to_time(sunrise_time_lst)
        sunset_time = SunPosition._day_fraction_to_time(sunset_time_lst)
        solar_noon_time = SunPosition._day_fraction_to_time(solar_noon_lst)
        
        return SunPositionData(
            elevation=solar_elevation_angle_deg,
            corrected_elevation=solar_elevation_corrected_deg,
            azimuth=solar_azimuth_angle_deg,
            sun_direction=sun_direction,
            sunrise_time=sunrise_time,
            sunset_time=sunset_time,
            solar_noon=solar_noon_time
        )
    
    @staticmethod
    def elevation_azimuth_to_direction(elevation: float, azimuth: float) -> tuple[float, float, float]:
        """
        Convert elevation and azimuth angles to a unit direction vector in Y-up coordinate system.
        
        Coordinate system:
        - Y is up (vertical)
        - Z is North (forward when azimuth = 0)
        - X is East (right when facing North)
        
        Args:
            elevation: Elevation angle in degrees (0 = horizon, 90 = zenith, negative = below horizon)
            azimuth: Azimuth angle in degrees (0 = North, 90 = East, 180 = South, 270 = West)
            
        Returns:
            Tuple (x, y, z) representing the normalized sun direction vector
        """
        elevation_rad = math.radians(elevation)
        azimuth_rad = math.radians(azimuth)
        
        cos_elevation = math.cos(elevation_rad)
        sin_elevation = math.sin(elevation_rad)
        
        # X = East component (sin(azimuth) when looking from above)
        x = cos_elevation * math.sin(azimuth_rad)
        # Y = Up component
        y = sin_elevation
        # Z = North component (cos(azimuth) when looking from above)  
        z = cos_elevation * math.cos(azimuth_rad)
        
        return (x, y, z)
    
    @staticmethod
    def _get_julian_day(year: int, month: int, day: int, hour: int, minute: int, second: int) -> float:
        """
        Calculate Julian Day number for given date and time.
        
        Julian Day is the continuous count of days since the beginning of the Julian Period
        (January 1, 4713 BC in the Julian calendar).
        
        Args:
            year, month, day: Date components
            hour, minute, second: Time components
            
        Returns:
            Julian Day number as a float
        """
        # Algorithm from Astronomical Algorithms by Jean Meeus
        if month <= 2:
            year -= 1
            month += 12
        
        a = int(year / 100)
        b = 2 - a + int(a / 4)
        
        jd = (
            int(365.25 * (year + 4716))
            + int(30.6001 * (month + 1))
            + day + b - 1524.5
        )
        
        # Add time of day as fraction
        jd += (hour + minute / 60.0 + second / 3600.0) / 24.0
        
        return jd
    
    @staticmethod
    def _day_fraction_to_time(day_fraction: float) -> time:
        """
        Convert a day fraction (0.0 - 1.0) to a time object.
        
        Args:
            day_fraction: Fraction of day (0.0 = midnight, 0.5 = noon, 1.0 = midnight next day)
            
        Returns:
            time object representing the time of day
        """
        # Handle overflow/underflow
        day_fraction = day_fraction % 1.0
        if day_fraction < 0:
            day_fraction += 1.0
        
        total_seconds = int(day_fraction * 24 * 60 * 60)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        # Handle edge case of exactly midnight
        if hours >= 24:
            hours = 0
        
        return time(hours, minutes, seconds)
    
    # Default location: Chengdu, China
    DEFAULT_LATITUDE = 30.5728
    DEFAULT_LONGITUDE = 104.0668
    
    @staticmethod
    def get_current_location() -> tuple[float, float]:
        """
        Get current geographic location based on IP address (blocking).
        
        Uses ip-api.com free service. Falls back to Chengdu, China coordinates
        if the request fails.
        
        Returns:
            Tuple (latitude, longitude) in degrees
        """
        try:
            import urllib.request
            import json
            
            # Use ip-api.com (free, no API key required)
            url = "http://ip-api.com/json/?fields=status,lat,lon"
            
            with urllib.request.urlopen(url, timeout=5) as response:
                data = json.loads(response.read().decode('utf-8'))
                
                if data.get('status') == 'success':
                    lat = data.get('lat')
                    lon = data.get('lon')
                    if lat is not None and lon is not None:
                        return (float(lat), float(lon))
                
                # API returned but with failure status
                print(f"[SunPosition] Location API returned failure, using default (Chengdu)")
                return (SunPosition.DEFAULT_LATITUDE, SunPosition.DEFAULT_LONGITUDE)
                
        except Exception as e:
            print(f"[SunPosition] Failed to get location: {e}, using default (Chengdu)")
            return (SunPosition.DEFAULT_LATITUDE, SunPosition.DEFAULT_LONGITUDE)
    
    @staticmethod
    def get_current_location_async(callback: callable) -> None:
        """
        Get current geographic location based on IP address (non-blocking).
        
        Immediately returns, then calls the callback with (latitude, longitude) 
        when the request completes. Falls back to Chengdu coordinates on failure.
        
        Args:
            callback: Function to call with (latitude, longitude) when complete
        """
        import threading
        
        def fetch_location():
            lat, lon = SunPosition.get_current_location()
            callback(lat, lon)
        
        thread = threading.Thread(target=fetch_location, daemon=True)
        thread.start()
    
    @staticmethod
    def get_local_timezone() -> float:
        """
        Get local timezone offset from UTC in hours.
        
        Returns:
            Timezone offset in hours (e.g., 8.0 for UTC+8)
        """
        import time as time_module
        # Get local timezone offset in seconds, convert to hours
        # Note: tm_gmtoff is seconds east of UTC
        if hasattr(time_module, 'timezone'):
            # time.timezone is seconds west of UTC for standard time
            # Negate to get hours east of UTC
            return -time_module.timezone / 3600.0
        return 0.0
    
    @staticmethod
    def calculate_now(
        latitude: float | None = None,
        longitude: float | None = None,
        timezone: float | None = None
    ) -> SunPositionData:
        """
        Calculate sun position for current time and location.
        
        If latitude/longitude are not provided, attempts to get current location
        from IP address (falls back to Chengdu, China if request fails).
        
        If timezone is not provided, uses the system's local timezone.
        
        Args:
            latitude: Geographic latitude in degrees (optional)
            longitude: Geographic longitude in degrees (optional)
            timezone: Timezone offset from UTC in hours (optional)
            
        Returns:
            SunPositionData containing elevation, azimuth, direction vector and times
        """
        # Get location if not provided
        if latitude is None or longitude is None:
            lat, lon = SunPosition.get_current_location()
            latitude = latitude if latitude is not None else lat
            longitude = longitude if longitude is not None else lon
        
        # Get timezone if not provided
        if timezone is None:
            timezone = SunPosition.get_local_timezone()
        
        # Get current local time
        now = datetime.now()
        
        return SunPosition.calculate(
            latitude=latitude,
            longitude=longitude,
            year=now.year,
            month=now.month,
            day=now.day,
            hour=now.hour,
            minute=now.minute,
            second=now.second,
            timezone=timezone,
            daylight_saving=False  # Python's datetime already accounts for DST
        )


# Example usage and test
if __name__ == "__main__":
    # Test with Montreal, Canada (from UE SunPosition test cases)
    print("=" * 60)
    print("Sun Position Calculator Test")
    print("=" * 60)
    
    # Test case: Montreal, Canada - December 21, 2017, 12:42 PM
    result = SunPosition.calculate(
        latitude=45.0,
        longitude=-73.0,
        year=2017, month=12, day=21,
        hour=12, minute=42, second=0,
        timezone=-5.0,
        daylight_saving=False
    )
    
    print(f"\nLocation: Montreal, Canada (45°N, 73°W)")
    print(f"Date/Time: 2017-12-21 12:42:00 (UTC-5)")
    print(f"\nResults:")
    print(f"  Elevation: {result.elevation:.3f}°")
    print(f"  Corrected Elevation: {result.corrected_elevation:.3f}°")
    print(f"  Azimuth: {result.azimuth:.3f}°")
    print(f"  Sun Direction (Y-up): ({result.sun_direction[0]:.4f}, {result.sun_direction[1]:.4f}, {result.sun_direction[2]:.4f})")
    print(f"\nAstronomical Times:")
    print(f"  Sunrise: {result.sunrise_time}")
    print(f"  Solar Noon: {result.solar_noon}")
    print(f"  Sunset: {result.sunset_time}")
    
    # Expected values from UE test: Azimuth=192.659, Elevation=20.556, CorrectedElevation=20.599
    print(f"\nExpected (from UE test):")
    print(f"  Elevation: 20.556°, Azimuth: 192.659°")
    
    # Test case 2: Sydney, Australia - December 21, 2017, 6:30 AM
    print("\n" + "=" * 60)
    result2 = SunPosition.calculate(
        latitude=-33.0,
        longitude=-151.0,  # Note: UE test uses negative for west
        year=2017, month=12, day=21,
        hour=6, minute=30, second=0,
        timezone=10.0,
        daylight_saving=False
    )
    
    print(f"\nLocation: Sydney, Australia (33°S, 151°E)")
    print(f"Date/Time: 2017-12-21 06:30:00 (UTC+10)")
    print(f"\nResults:")
    print(f"  Elevation: {result2.elevation:.3f}°")
    print(f"  Azimuth: {result2.azimuth:.3f}°")
    print(f"  Sun Direction (Y-up): ({result2.sun_direction[0]:.4f}, {result2.sun_direction[1]:.4f}, {result2.sun_direction[2]:.4f})")
    print(f"\nAstronomical Times:")
    print(f"  Sunrise: {result2.sunrise_time}")
    print(f"  Solar Noon: {result2.solar_noon}")
    print(f"  Sunset: {result2.sunset_time}")
    
    # Expected: Azimuth=70.526, Elevation=67.682
    print(f"\nExpected (from UE test):")
    print(f"  Elevation: 67.682°, Azimuth: 70.526°")
    
    # Test case 3: Current location and time
    print("\n" + "=" * 60)
    print("Current Location & Time Test")
    print("=" * 60)
    
    lat, lon = SunPosition.get_current_location()
    tz = SunPosition.get_local_timezone()
    print(f"\nDetected Location: ({lat:.4f}°, {lon:.4f}°)")
    print(f"Detected Timezone: UTC{'+' if tz >= 0 else ''}{tz:.1f}")
    
    result3 = SunPosition.calculate_now()
    now = datetime.now()
    print(f"Current Time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nResults:")
    print(f"  Elevation: {result3.elevation:.3f}°")
    print(f"  Azimuth: {result3.azimuth:.3f}°")
    print(f"  Sun Direction (Y-up): ({result3.sun_direction[0]:.4f}, {result3.sun_direction[1]:.4f}, {result3.sun_direction[2]:.4f})")
    print(f"\nAstronomical Times:")
    print(f"  Sunrise: {result3.sunrise_time}")
    print(f"  Solar Noon: {result3.solar_noon}")
    print(f"  Sunset: {result3.sunset_time}")

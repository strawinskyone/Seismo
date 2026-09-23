"""
Агрегатор импортов виджетов сейсмостанции.
"""
from map_widget import MapWidget, SeismicEvent
from osc_widget import OscilloscopeWidget
from water_alarm import WaterAlarmWidget

__all__ = ['MapWidget', 'SeismicEvent', 'OscilloscopeWidget', 'WaterAlarmWidget']
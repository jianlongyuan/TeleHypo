# TeleHypo
Hypocenter location method for teleseismic earthquakes

Framework:
    0_Run_TeleHypo.py
	|sub1_FetchData.py
	|sub2_CalSignalToNoiseRatio.py
	|sub3_FetchInventory.py
	|sub4_SelectAzimuthalStations.py
	|sub5_PreliminaryLocation.py
	|sub6_PreciseLocation.py

Quickstart:
    All input parameters can be seen in the file "SETTINGS.txt",
    Run the python scripts (The Chile earthquake case used in paper):
        0_Run_TeleHypo.py


  Output results can be seen at:
        ./catalog_GCMT_2010-03-04_2010-05-13_Mw6.0-8.0_50-300km/2010-03-04-22-39-29/ssnapresults
        ./catalog_GCMT_2010-03-04_2010-05-13_Mw6.0-8.0_50-300km/2010-03-04-22-39-29/DSA_results

  
  Plot location results:
        tool_plot_SNR.py
        tool_plot_brightness function.py
        tool_plot_DSA_depth_solution.py
        tool_plot_solutions_comparison.py


Please cite:

Tan, F., Kao, H., Nissen, E., and Eaton, D. (2019). Seismicity-scanning based on
navigated automatic phase-picking. Journal of Geophysical Research: Solid Earth 124, 3802–3818

Yuan, J., Kao, H., and Yu, J. (2020). Depth-scanning algorithm: Accurate, automatic,
and efficient determination of focal depths for local and regional earthquakes.
Journal of Geophysical Research: Solid Earth 125, e2020JB019430
    
Jianlong Yuan, Huilian Ma, Jiashun Yu, Zixuan Liu and Shaojie Zhang. (2025). An approach 
for teleseismic location by automatically matching depth phase. Front. Earth Sci. (Under revirew)
  

Any questions or advices? Please contact at:
    jianlongyuan@cdut.edu.cn (Jianlong Yuan)
    1334631943@qq.com (Huilian Ma)
    j.yu@cdut.edu.cn  (Jiashun Yu)
    2751017165@qq.com (Zixuan Liu)
    1716136870@qq.com (Shaojie Zhang)
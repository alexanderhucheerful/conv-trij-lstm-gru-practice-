begin
hx=addfile("HadISST_sst.nc", "r")
sst1=hx->sst
lat=hx->latitude
lon=hx->longitude
sst=new((/1788,180,360/), "float")
sst=sst1(0:1787,:,:)
sstwinter=month_to_season(sst, "DJF")
sstsummer=month_to_season(sst, "JJA")
sstspring=month_to_season(sst, "MAM")
sstautumn=month_to_season(sst, "SON")

sstspringbl=dim_stddev_n_Wrap(sstspring, 0)
sstsummerbl=dim_stddev_n_Wrap(sstsummer, 0)
sstautumnbl=dim_stddev_n_Wrap(sstautumn, 0)
sstwinterbl=dim_stddev_n_Wrap(sstwinter, 0)
print(sstwinterbl)

plot = new(4, graphic)
wks_type = "png"
wks_type@wkWidth = 2400
wks_type@wkHeight =2400
wks = gsn_open_wks(wks_type,"blall")
;gsn_define_colormap(wks,"BlueDarkRed18")
res                       = True
;res@gsnAddCyclic = False;huazhongguoquyujiashang
res@gsnMaximize           = True
res@gsnDraw               = False
res@gsnFrame              = False
res@tmXBLabelDeltaF = -0.8
res@tmYLLabelDeltaF = -0.8
res@tmXBLabelFont = 25   ;times roman
res@tmYLLabelFont = 25
res@mpFillOn              = True     ; turn on color fill
res@mpCenterLonF   =0;180
res@mpMinLonF=0
res@mpMaxLonF=360
res@mpMinLatF=-90
res@mpMaxLatF=90

res@tmYLMinorValues= (/-90,-85,-80,-75,-70,-65,-60,-55,-50,-45,-40,-35,-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90/)
res@tmXBMinorValues= (/0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360/)

res@mpDataSetName         = "Earth..4"
res@mpDataBaseVersion     = "MediumRes"
res@mpOutlineOn          = True
;res@mpAreaMaskingOn       = True;to only draw china area
;res@mpMaskAreaSpecifiers  = (/"China","Taiwan"/);to only draw china area
;res@mpOutlineSpecifiers   = (/"China:states"/);(/"China","China:states"/);Provinces"/)
res@mpOutlineSpecifiers = (/"China:states","India","Nepal","Bhutan","Bangladesh","Myanmar", "Thailand","Cambodia","Vietnam","Taiwan"/)
res@mpLandFillColor            = "white"
res@mpInlandWaterFillColor     = "white"
res@mpOceanFillColor           = "white"
;res@mpFillBoundarySets         = "NoBoundaries"
;res@mpOutlineBoundarySets      = "NoBoundaries"
res@cnLevelSelectionMode = "ExplicitLevels"
;res@cnLevels             = (/18,21,24,27,22,23,24,25,26,27,28/)
res@cnLevels             = (/0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1/)
;res@cnLevels             = (/-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5/);(/-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1/);(/-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5/)
res@cnExplicitLineLabelsOn = True
res@cnLineLabelStrings   = (/0,1,2,3,4,5,6,7,8,9,10/)
;res@cnLineLabelStrings   = (/-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5/);(/-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1/);(/-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5/)
res@mpNationalLineColor        = "black"
res@mpProvincialLineColor      = "black"
res@mpGeophysicalLineColor     = "black"
res@mpNationalLineThicknessF   = 2
res@mpProvincialLineThicknessF = 1

res@cnFillOn              = True
;res@cnFillDrawOrder       = "PreDraw";to only draw china area
res@cnLineLabelsOn        = False
res@cnLinesOn             = False    ; turn of contour lines
;res@cnLevelSpacingF       = 0.5      ; contour spacing
res@lbLabelAutoStride          = True
res@lbLabelFont =25
;res@pmLabelBarWidthF     =0.6
res@pmLabelBarHeightF     =0.08
res@pmLabelBarOrthogonalPosF = .15
;res@pmTickMarkDisplayMode = "Always"

res@lbBoxLinesOn          =False
;colors=(/2,3,4,5,6,7,14,15,16,17,18,19/);(/2,3,4,5,6,7,8,9,10,11,12,13/)
;res@cnFillColors         = colors
res@gsnLeftString=" "
res@gsnRightString=" "
res@tmYLMode       = "Explicit"
;res@tmYLLabelsOn = True
res@tmYLMinorOn    = True
res@tmXBMode       = "Explicit"
res@tmXBMinorOn    = True
res@tmYLValues     = (/-90,-60,-30,0,30,60,90/);(/15,25,35,45,55/)
res@tmXBValues     = (/0,60,120,180,240,300,360/);(/75,85,95,105,115,125,135/)
res@tmYLLabels     = (/"90S","60S","30S","EQ","30N","60N","90N"/);(/"15N","25N","35N","45N","55N"/)
res@tmXBLabels     = (/"0","60E","120E","180","120W","60W","0"/);(/"75E","85E","95E","105E","115E","125E","135E"/)
res@tmYROn= False
res@tmXTOn= False
res@tmYLMinorPerMajor= 2
res@tmXBMinorPerMajor =4
res@tmBorderThicknessF          = 10.
res@tmXBMajorThicknessF         = 8.
res@tmYLMajorThicknessF         = 8.
res@tmXBMinorThicknessF         = 5.
res@tmYLMinorThicknessF         = 5.
res@tmXBLabelFontHeightF = 0.013
res@tmYLLabelFontHeightF  = 0.013
res@tmXBLabelDeltaF = -0.8
res@tmYLLabelDeltaF = -0.8
res@tmYLLabelFont = 25
res@tmXBLabelFont = 25
res@lbLabelFont = 25
res@cnLineLabelFont=25


res0 = res
res1= res
res2 = res
res3 = res
res0@gsnCenterString = "spring variation"
res1@gsnCenterString = "summer variation"
res2@gsnCenterString = "autumn variation"
res3@gsnCenterString = "winter variation"

plot(0) = gsn_csm_contour_map(wks,sstspringbl,res0)         ; create a default plot
plot(1) = gsn_csm_contour_map(wks,sstsummerbl,res1) 
plot(2) = gsn_csm_contour_map(wks,sstautumnbl,res2) 
plot(3) = gsn_csm_contour_map(wks,sstwinterbl,res3) 

pnlres = True
pnlres@gsnPanelMainString =  "sst season variation "
gsn_panel(wks,plot,(/2,2/),pnlres)

end
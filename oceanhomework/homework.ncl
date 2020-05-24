begin
nino=asciiread("ninojp-index.txt", (/151,13/), "float")
t2m=addfile("DJF_T2m_ERA5_0.25_0.25.nc", "r")
pre=addfile("JJA_tol_preci_ERA5_0.25_0.25.nc", "r")

t2mdj1=short2flt(t2m->t2m_0001)
t2mdj=t2mdj1(0:39,:,:)
;t2mdj=dim_standardize(t2mdj2,0)
prexj1=short2flt(pre->tp)
prexj=dim_standardize(prexj1(:,0,:,:),0)
londj=t2m->longitude
latdj=t2m->latitude
lonxj=pre->longitude
latxj=pre->latitude

ninojja=new((/41,3/), "float")
ninodjf=new((/40,3/), "float")

do i=109,149
    ninojja(i-109,0)=nino(i,6)
    ninojja(i-109,1)=nino(i,7)
    ninojja(i-109,2)=nino(i,8)
end do
delete(i)

do i=109,148
    ninodjf(i-109,0)=nino(i,12)
    ninodjf(i-109,1)=nino(i+1,1)
    ninodjf(i-109,2)=nino(i+1,2)
end do
delete(i)

ninoxj=new((/41/),"float")
ninodj=new((/40/),"float")
ninoxj3dvar=new((/41,361,1440/),"float")
ninodj3dvar=new((/40,721,1440/),"float")

ninoxj=dim_avg_n(ninojja, 1)
ninodj=dim_avg_n(ninodjf, 1)

bxj=regCoef_n(ninoxj, prexj, 0, 0)
bdj=regCoef_n(ninodj, t2mdj, 0, 0)

bxj!1="lon"
bxj@long_name = "lon"
bxj@unils = "degress_east"
bxj&lon = lonxj
bxj!0="lat"
bxj@long_name = "lat"
bxj@unils = "degress_north"
bxj&lat = latxj

bdj!1="lon"
bdj@long_name = "lon"
bdj@unils = "degress_east"
bdj&lon = londj
bdj!0="lat"
bdj@long_name = "lat"
bdj@unils = "degress_north"
bdj&lat = latdj

wks_type = "png"
  wks_type@wkWidth = 2400
  wks_type@wkHeight =2400
wks = gsn_open_wks(wks_type,"bdj");"tsera5-chazhi2016070100")
wks1 = gsn_open_wks(wks_type,"t2mzd")
;gsn_define_colormap(wks,)
gsn_define_colormap(wks1,"BlueRed")
  res                       = True
  res@gsnAddCyclic = False;huazhongguoquyujiashang
  res@gsnMaximize           = True
  res@gsnDraw               = False
  res@gsnFrame              = False
res@tmXBLabelDeltaF = -0.8
res@tmYLLabelDeltaF = -0.8
res@tmXBLabelFont = 25   ;times roman
res@tmYLLabelFont = 25
  res@mpFillOn              = True     ; turn on color fill
  res@mpCenterLonF   =0;180
  res@mpMinLonF=75
  res@mpMaxLonF=135;150
  res@mpMinLatF=15
  res@mpMaxLatF=55

  res@tmYLMinorValues= (/-90,-85,-80,-75,-70,-65,-60,-55,-50,-45,-40,-35,-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90/)
  res@tmXBMinorValues= (/0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360/)
  ;res@tmYLMinorValues= (/30,40/)
  ;res@tmXBMinorValues= (/ /)


  res@mpDataSetName         = "Earth..4"
  res@mpDataBaseVersion     = "MediumRes"
  res@mpOutlineOn          = True
  res@mpAreaMaskingOn       = True;to only draw china area
  res@mpMaskAreaSpecifiers  = (/"China","Taiwan"/);to only draw china area
  ;res@mpOutlineSpecifiers   = (/"China:states"/);(/"China","China:states"/);Provinces"/)
res@mpOutlineSpecifiers = (/"China:states","India","Nepal","Bhutan","Bangladesh","Myanmar", "Thailand","Cambodia","Vietnam","Taiwan"/)
  res@mpLandFillColor            = "white"
  res@mpInlandWaterFillColor     = "white"
  res@mpOceanFillColor           = "white"
  ;res@mpFillBoundarySets         = "NoBoundaries"
  ;res@mpOutlineBoundarySets      = "NoBoundaries"
  res@cnLevelSelectionMode = "ExplicitLevels"
  ;res@cnLevels             = (/18,21,24,27,22,23,24,25,26,27,28/)
  ;res@cnLevels             = (/5,8,11,14,17,20,23,26,29,32,35/)
  res@cnLevels             = (/-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5/);(/-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1/);(/-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5/)
  res@cnExplicitLineLabelsOn = True
  ;res@cnLineLabelStrings   = (/5,8,11,14,17,20,23,26,29,32,35/)
  res@cnLineLabelStrings   = (/-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5/);(/-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1/);(/-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5/)
    res@mpNationalLineColor        = "black"
  res@mpProvincialLineColor      = "black"
  res@mpGeophysicalLineColor     = "black"
  res@mpNationalLineThicknessF   = 2
  res@mpProvincialLineThicknessF = 1

  res@cnFillOn              = True
  res@cnFillDrawOrder       = "PreDraw";to only draw china area
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
  colors=(/2,3,4,5,6,7,14,15,16,17,18,19/);(/2,3,4,5,6,7,8,9,10,11,12,13/)
  ;res@cnFillColors         = colors
  res@gsnLeftString=" "
  res@gsnRightString=" "
  res@tmYLMode       = "Explicit"
  ;res@tmYLLabelsOn = True
  res@tmYLMinorOn    = True
  res@tmXBMode       = "Explicit"
  res@tmXBMinorOn    = True
  res@tmYLValues     = (/15,25,35,45,55/);(/15,25,35,45,55/)
  res@tmXBValues     = (/75,85,95,105,115,125,135/);(/75,85,95,105,115,125,135/)
  res@tmYLLabels     = (/"15N","25N","35N","45N","55N"/);(/"15N","25N","35N","45N","55N"/)
  res@tmXBLabels     = (/"75E","85E","95E","105E","115E","125E","135E"/);(/"75E","85E","95E","105E","115E","125E","135E"/)
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




  plot = gsn_csm_contour_map(wks,bdj,res)         ; create a default plot
    draw(plot)
  frame(wks)


end
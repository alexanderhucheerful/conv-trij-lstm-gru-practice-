begin
nino = asciiread("ninojp-index.txt", -1, "string")
nino := nino(1:151,:)
year = toint(str_get_field(nino,1, "    "))
Dec = tofloat(str_get_field(nino,13, "    "))
Jan = tofloat(str_get_field(nino, 2, "    "))
Feb = tofloat(str_get_field(nino, 3, "    "))
nino_DJF = (Dec(0:149) + Jan(1:150) + Feb(1:150))/3

T = addfile("DJF_T2m_ERA5_0.25*0.25.nc", "r")
Prec = addfile("JJA_tol_preci_ERA5_0.25*0.25.nc", "r")
t = short2flt(T->t2m_0001)
prec = short2flt(Prec->tp(:,0,:,:))
;截取需要的时间段进行计算
;nino
countTime_1 = ind(year.le.2010.and.year.ge.1981)
;t
YYYY = cd_calendar(t&time,0)
yyyy = YYYY(:,0)
countTime_2 = ind(yyyy.le.2011.and.yyyy.ge.1982)
delete([/YYYY,yyyy/])
;prec
YYYY = cd_calendar(prec&time,0)
yyyy = YYYY(:,0)
countTime_3 = ind(yyyy.le.2011.and.yyyy.ge.1982)
delete([/YYYY,yyyy/])
;count r
r1 = t(0,:,:)
r2 = prec(0,:,:)
r1 = escorc_n(nino_DJF(countTime_1),t(countTime_2,:,:),0,0)  
r2 = escorc_n(nino_DJF(countTime_1),prec(countTime_3,:,:),0,0)

;draw 
plot = new(2, graphic)
wks = gsn_open_wks("png","work1")
gsn_define_colormap(wks,"amwg256")

res = True
res@gsnDraw   = False
res@gsnFrame  = False
res@mpMaxLatF = 60
res@mpMinLatF = 5
res@mpMaxLonF = 140
res@mpMinLonF = 70
res@mpDataBaseVersion = "Ncarg4_1"
res@mpDataSetName     = "Earth..4"
res@mpFillOn          = True
res@mpOceanFillColor       = 0
res@mpInlandWaterFillColor = 0
res@mpLandFillColor        = 0
res@mpAreaMaskingOn        = True
res@mpMaskAreaSpecifiers   = (/"China","Taiwan","Disputed area between India and China"/)
res@mpOutlineBoundarySets = "NoBoundaries"
res@mpFillDrawOrder = "PostDraw"
res@gsnLeftString  = ""
res@gsnRightString = ""

res@cnFillOn  = True
res@cnInfoLabelOn  = False
res@cnLineLabelsOn = False
res@lbLabelBarOn   = False  
res@cnLevelSelectionMode = "ManualLevels"
res@cnMaxLevelValF = 0.8
res@cnMinLevelValF = -0.8
res@cnLevelSpacingF = 0.2
; res@cnFillColors = (/66,82,98,114,130,146,162,178,194/)
res@cnLinesOn = False
res@cnFillDrawOrder = "PreDraw"
plot(0) = gsn_csm_contour_map(wks,r1, res)
plot(1) = gsn_csm_contour_map(wks,r2, res)


pres = True
pres@gsnLeftString  = ""
pres@gsnRightString = ""
gsn_panel(wks,plot, (/1,2/), pres)


end 
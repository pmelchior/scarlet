Search.setIndex({docnames:["api_docs","bbox","blend","cache","changes","component","display","index","install","interpolation","observation","operator","psf","quickstart","resampling","source","tutorials","tutorials/multiresolution","tutorials/point_source","update","user_docs"],envversion:{"sphinx.domains.c":1,"sphinx.domains.changeset":1,"sphinx.domains.cpp":1,"sphinx.domains.javascript":1,"sphinx.domains.math":2,"sphinx.domains.python":1,"sphinx.domains.rst":1,"sphinx.domains.std":1,"sphinx.ext.intersphinx":1,"sphinx.ext.viewcode":1,nbsphinx:1,sphinx:55},filenames:["api_docs.rst","bbox.ipynb","blend.ipynb","cache.ipynb","changes.rst","component.ipynb","display.ipynb","index.rst","install.rst","interpolation.ipynb","observation.ipynb","operator.ipynb","psf.ipynb","quickstart.ipynb","resampling.ipynb","source.ipynb","tutorials.rst","tutorials/multiresolution.ipynb","tutorials/point_source.ipynb","update.ipynb","user_docs.ipynb"],objects:{"scarlet.bbox":{BoundingBox:[1,1,1,""],flux_at_edge:[1,3,1,""],resize:[1,3,1,""],trim:[1,3,1,""]},"scarlet.bbox.BoundingBox":{height:[1,2,1,""],shape:[1,2,1,""],slices:[1,2,1,""],width:[1,2,1,""]},"scarlet.blend":{Blend:[2,1,1,""]},"scarlet.blend.Blend":{converged:[2,2,1,""],fit:[2,4,1,""],it:[2,2,1,""],sources:[2,2,1,""]},"scarlet.cache":{Cache:[3,1,1,""]},"scarlet.component":{BlendFlag:[5,1,1,""],Component:[5,1,1,""],ComponentTree:[5,1,1,""],Prior:[5,1,1,""]},"scarlet.component.Component":{backward_prior:[5,4,1,""],coord:[5,2,1,""],get_flux:[5,4,1,""],get_model:[5,4,1,""],morph:[5,2,1,""],sed:[5,2,1,""],shape:[5,2,1,""],update:[5,4,1,""]},"scarlet.component.ComponentTree":{B:[5,2,1,""],K:[5,2,1,""],Nx:[5,2,1,""],Ny:[5,2,1,""],components:[5,2,1,""],coord:[5,2,1,""],get_flux:[5,4,1,""],get_model:[5,4,1,""],n_components:[5,2,1,""],n_nodes:[5,2,1,""],nodes:[5,2,1,""],update:[5,4,1,""]},"scarlet.component.Prior":{compute_grad:[5,4,1,""]},"scarlet.display":{AsinhPercentileNorm:[6,1,1,""],LinearPercentileNorm:[6,1,1,""],img_to_channel:[6,3,1,""],img_to_rgb:[6,3,1,""]},"scarlet.interpolation":{bilinear:[9,3,1,""],catmull_rom:[9,3,1,""],common_projections:[9,3,1,""],cubic_spline:[9,3,1,""],fft_convolve:[9,3,1,""],fft_resample:[9,3,1,""],get_common_padding:[9,3,1,""],get_projection_slices:[9,3,1,""],get_separable_kernel:[9,3,1,""],lanczos:[9,3,1,""],mitchel_netravali:[9,3,1,""],project_image:[9,3,1,""],sinc2D:[9,3,1,""],sinc_interp:[9,3,1,""]},"scarlet.observation":{LowResObservation:[10,1,1,""],Observation:[10,1,1,""],Scene:[10,1,1,""]},"scarlet.observation.LowResObservation":{get_loss:[10,4,1,""],get_model:[10,4,1,""],get_model_image:[10,4,1,""],match:[10,4,1,""]},"scarlet.observation.Observation":{get_loss:[10,4,1,""],get_model:[10,4,1,""],get_scene:[10,4,1,""]},"scarlet.observation.Scene":{B:[10,2,1,""],Nx:[10,2,1,""],Ny:[10,2,1,""],get_pixel:[10,4,1,""],shape:[10,2,1,""]},"scarlet.operator":{diagonalizeArray:[11,3,1,""],diagonalsToSparse:[11,3,1,""],find_Q:[11,3,1,""],find_relevant_dim:[11,3,1,""],getOffsets:[11,3,1,""],getRadialMonotonicOp:[11,3,1,""],getRadialMonotonicWeights:[11,3,1,""],proj:[11,3,1,""],proj_dist:[11,3,1,""],project_disk_sed:[11,3,1,""],project_disk_sed_mean:[11,3,1,""],prox_center_on:[11,3,1,""],prox_cone:[11,3,1,""],prox_max_unity:[11,3,1,""],prox_sdss_symmetry:[11,3,1,""],prox_sed_on:[11,3,1,""],prox_soft_symmetry:[11,3,1,""],prox_strict_monotonic:[11,3,1,""],prox_uncentered_symmetry:[11,3,1,""],proximal_disk_sed:[11,3,1,""],sort_by_radius:[11,3,1,""],uncentered_operator:[11,3,1,""],use_relevant_dim:[11,3,1,""]},"scarlet.psf":{double_gaussian:[12,3,1,""],fit_target_psf:[12,3,1,""],gaussian:[12,3,1,""],moffat:[12,3,1,""]},"scarlet.resampling":{conv2D_fft:[14,3,1,""],linorm2D:[14,3,1,""],make_operator:[14,3,1,""],match_patches:[14,3,1,""],match_psfs:[14,3,1,""]},"scarlet.source":{CombinedExtendedSource:[15,1,1,""],ExtendedSource:[15,1,1,""],MultiComponentSource:[15,1,1,""],PointSource:[15,1,1,""],SourceInitError:[15,5,1,""],build_detection_coadd:[15,3,1,""],get_best_fit_seds:[15,3,1,""],get_pixel_sed:[15,3,1,""],get_scene_sed:[15,3,1,""],init_combined_extended_source:[15,3,1,""],init_extended_source:[15,3,1,""],init_multicomponent_source:[15,3,1,""]},"scarlet.source.PointSource":{update:[15,4,1,""]},"scarlet.update":{fit_pixel_center:[19,3,1,""],monotonic:[19,3,1,""],normalized:[19,3,1,""],positive:[19,3,1,""],positive_morph:[19,3,1,""],positive_sed:[19,3,1,""],sparse_l0:[19,3,1,""],sparse_l1:[19,3,1,""],symmetric:[19,3,1,""],threshold:[19,3,1,""],translation:[19,3,1,""]},scarlet:{bbox:[1,0,0,"-"],blend:[2,0,0,"-"],cache:[3,0,0,"-"],component:[5,0,0,"-"],display:[6,0,0,"-"],interpolation:[9,0,0,"-"],observation:[10,0,0,"-"],operator:[11,0,0,"-"],psf:[12,0,0,"-"],resampling:[14,0,0,"-"],source:[15,0,0,"-"],update:[19,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","attribute","Python attribute"],"3":["py","function","Python function"],"4":["py","method","Python method"],"5":["py","exception","Python exception"]},objtypes:{"0":"py:module","1":"py:class","2":"py:attribute","3":"py:function","4":"py:method","5":"py:exception"},terms:{"1st":9,"2nd":9,"2x2":9,"2xm":9,"2xn":9,"3x3":19,"4750118475607095e":17,"8xn":11,"break":[4,7,11,19,20],"byte":17,"case":[4,7,11,13,17,18,19,20],"class":[1,2,3,4,5,6,7,10,11,13,15,17,20],"default":[4,5,6,7,12,15,20],"final":[17,20],"float":[1,2,6,9,10,11,14,15],"function":[2,4,5,7,8,9,10,11,12,13,14,15,19,20],"import":[4,7,13,17,18],"int":[1,2,9,10,11,14,15,17,20],"long":18,"new":[1,9,10,11],"public":8,"return":[1,2,4,5,6,7,9,10,11,13,14,15,17,18,19,20],"short":20,"super":20,"switch":8,"true":[11,13,15,17,18,19,20],"try":[13,20],"while":[4,5,7,9,11,13,15,18,20],But:20,For:[1,2,6,11,13,15,17,18,19,20],One:20,RMS:[15,17,20],THAT:8,The:[1,2,4,5,7,8,9,10,11,13,15,17,18,19,20],Then:[8,13,17],There:[5,20],These:[9,10,20],Use:[5,9,18,19,20],Uses:[11,15],Using:[11,13,18,20],WCS:[10,13,14,17],With:20,__getitem__:5,__init__:[5,15,20],_build:8,_coord_lr:17,_init_rgb:17,_model:[13,17,18,20],_morph:20,_parent:20,_rgb:20,_sed:20,abl:20,about:[9,13,19,20],abov:[1,8,13,19,20],abs:18,absorpt:20,abus:20,acceler:[4,7],accept:[8,20],access:[4,7,20],accur:[11,20],act:[5,11],add:[1,20],add_subplot:[13,18,20],added:[4,7,20],adding:20,addit:[5,10,11,20],address:18,adequ:20,adjust:[18,20],advanc:20,advantag:20,advers:[13,20],advis:1,affect:[13,20],after:[11,18,20],again:20,ahead:15,algorithm:[2,4,7,8,11,18,20],align:11,all:[1,3,4,5,7,8,10,11,13,15,17,18,19,20],all_param:12,allow:[1,4,6,7,8,11,15,18,19,20],along:20,alpha:[12,20],alreadi:8,also:[13,17,18,20],although:13,alwai:[1,4,7,11,20],amount:[9,11,18,19],amplitud:[12,20],analyt:20,angl:11,ani:[1,4,6,7,9,10,13,15,20],anoth:[8,14,20],anp:[13,18,20],anywai:20,anywher:[4,7],api:20,appear:11,append:[13,18,20],appli:[4,7,11,17,20],apppli:17,approach:20,appropri:[11,18,20],approxim:[2,4,7,20],approximate_l:[2,20],arang:[13,20],arcsinh:[17,20],arg:20,argument:[5,9,12,15,17,19,20],aris:18,arisen:20,arr:[11,13,18,20],arrai:[1,2,5,6,9,10,11,12,13,14,15,17,18,20],array_lik:6,artifact:[10,18],asinh:[17,18,20],asinhmap:[13,17,18,20],asinhpercentilenorm:6,associ:10,assum:[9,15,20],assumpt:20,astrophys:[7,20],astropi:[4,7,8,13,17,18,20],astyp:17,attach:[5,15],attempt:20,attribut:[1,2,5,10,15,20],autograd:[4,7,8,13,18,20],automat:[8,20],avail:[8,20],averag:[11,17,20],avoid:[11,18],awai:[4,7],axes:[13,17,20],axi:[13,17,20],background:[15,17,18,20],backround:15,backward_prior:5,band:[4,5,6,7,10,11,13,15,17,18,20],bare:[13,20],base:[4,5,7,8,9,11,13,14,20],basic:11,bbox:[0,7,19,20],becaus:[4,7,11,15,17,18,20],been:[4,7,18,20],befor:[9,15,17,20],begin:20,behavior:[4,7,20],behind:8,being:[5,11],belong:17,below:[5,13,19,20],benefit:20,best:[4,7,15,20],beta:[12,20],better:[11,13,17,18,20],between:[9,11,14,15,20],beyond:[11,20],bg_cutoff:15,bg_rm:[13,15,17,18,20],bg_rms_hsc:17,bg_rms_hst:17,big:20,bilinear:[4,7,9],binari:[4,7],bit:[11,17,20],bkg:17,black:20,blend:[0,4,5,7,12,13,16,19],blendflag:[4,5,7],bluer:11,bool:[1,2,5,11,15,19],both:[5,11,17,18,19,20],bottom:[1,9,17],bound:[0,4,7,9,13,17,20],boundingbox:[1,4,7],box:[0,4,7,9,11,13,17,20],boyd:20,branch:[8,20],briefli:20,bright:[19,20],brighter:20,brightest:20,broken:[8,20],bug:17,build:[4,5,7,9,10,11,12,13,14,15,20],build_detection_coadd:15,build_ext:8,built:20,buldg:5,bulg:[11,18,20],bulge_s:11,byteswap:17,cach:[0,7],calcualt:17,calcul:[4,5,7,8,9,10,11,13,15,17,18,20],call:[4,5,7,20],cam:17,camera:20,can:[3,4,5,7,9,11,13,15,17,18,20],cannot:[13,18],care:18,cast:17,catalog:[17,18,20],catalog_hsc:17,catalog_hst:17,cataog:13,catmull_rom:9,caus:[4,7,18,20],caution:5,caveat:20,cell:20,center:[4,5,7,9,11,14,15,18,19],center_histori:20,center_step:15,central:[11,18,19,20],centroid:[4,7,20],certain:20,chang:[8,18,20],channel:[6,18,20],characterist:10,check:[1,2,3,7,11],checkout:8,children:20,choos:[13,18],chosen:3,circular:[12,20],clarifi:[4,7],clone:8,closer:[11,20],cluster:20,cmap:[13,17,18,20],coadd:[15,20],code:[1,4,7,8,13,18,20],collect:[2,4,5,7,12,20],color:[4,6,7,11,13,17,18,20],colorbar:17,colormap:[13,17,18,20],column:[11,20],com:[8,13],combett:20,combin:[11,15,20],combinedextendedsourc:[4,7,15,17],come:20,command:8,commit:[4,7,8],common:[9,13,20],common_project:9,commonli:20,comp:20,compar:[10,11,17,20],comparison:[11,20],compil:8,complet:[4,7,11,18],complex:[3,5,17,20],complexwarn:17,complic:[4,7,20],compon:[0,1,2,4,7,9,11,13,15,19],component_kwarg:15,componentlist:[4,7],componenttre:[2,4,5,7,15,20],compris:2,comput:[10,14],compute_grad:5,conatain:20,concaten:17,conclud:20,condit:20,cone:11,config:[4,7],confus:20,consequ:20,consider:17,consist:[17,18,19,20],constant:[2,4,5,7,13,14,18,20],constantli:20,constrain:[11,15,20],constraint:[4,5,7,9,13,15],construct:[5,11],consum:[3,17],contain:[1,2,3,4,7,9,10,11,12,13,15,17,19,20],contamin:20,content:3,continu:[13,20],control:20,conv2d_fft:14,conveni:[1,6,15,20],convent:[3,19,20],converg:[2,5,7,11,13,18],convergence_hist:20,convert:[6,11,17,18,20],convolut:[4,7,9,14,20],convolv:[9,10,14,18,20],convov:9,coord:[5,11,12,13,15,20],coord_hr:9,coord_lr:[9,14],coordin:[5,9,10,11,13,14,15,17,20],coordlr_over_hr:14,coordlr_over_lr:14,copi:[1,8],corner:9,correct:[4,7,17],correctli:[4,7],correspond:[6,13,20],cos:17,could:[11,13,20],cours:20,cover:20,cpu:18,crank:20,creat:[1,3,4,5,7,9,11,14,15,16,20],crowd:[18,20],crval:17,cube:[5,6,10,20],cubic:[4,7,9],cubic_splin:9,cubix:9,curiou:20,current:[1,4,7,8,10,20],curv:[10,13],custom:[11,12,20],cut_hsc:17,cut_hst:17,cutoff:[4,7,15,19],cyan:20,data:[2,5,6,7,8,10,11,16,18],data_hsc:17,data_hst:17,dataset:[14,17],debend:[4,7],deblend:[4,6,7,9,13,16,18,20],dec:17,deconvolut:18,deconvolv:18,decreas:[15,19,20],deeper:20,def:[13,17,20],defin:[1,4,7,10,11,15,17,18,20],definit:18,degeneraci:[4,7,19,20],degre:20,delay_thresh:15,demand:20,demonstr:18,depend:[8,18],deprec:[13,18,20],depth:18,deriv:[13,20],desc:18,describ:[11,18,20],descript:[15,19,20],design:17,desir:[15,19,20],detail:[9,11,18,20],detect:[5,13,15,17,18,20],detector:20,determin:[1,4,7,18,19,20],detet:20,develop:20,diag:11,diagon:11,diagonalizearrai:11,diagonalstospars:11,dict:[9,15],dictionari:1,did:20,didx:11,differ:[1,2,4,7,9,10,11,13,14,15,17,18,20],differenti:5,difficult:20,diffus:19,dimens:[6,9,11],dimension:11,dipol:[11,18],direct:[5,9,10,15,19,20],directli:20,directori:8,disadvantag:20,discard:17,discontinu:11,discuss:[17,20],disk:[5,11,20],disk_s:11,displai:[0,4,7,8,16],display_sourc:20,distanc:11,distinguish:20,distribut:15,diverg:[4,7],divid:20,doc:[4,7],docstr:8,document:[7,20],doe:[1,8,17,19,20],doesn:2,don:[8,13,17,18,20],done:[4,7,20],dot:[17,20],double_gaussian:12,download:8,drop:[4,7],dtype:[6,10,11,13,20],due:[4,5,7,10,20],dure:[5,8,15,20],dust:20,e_rel:[2,13,17,20],each:[2,4,5,7,10,11,12,13,15,17,18,19,20],earli:13,easier:[4,7,20],easili:20,edg:[1,4,5,7,9,20],edge_pixel:5,effect:[11,18,20],effici:20,eigen:8,either:[13,18,19,20],element:[11,20],elif:20,els:[13,17,18,20],emit:20,encod:10,encount:5,end:13,endian:17,enforc:[11,15,20],enough:[1,20],ensur:11,entir:[4,7,13,17,19,20],entri:1,enumer:[13,17,18,20],epoch:[7,20],equal:20,equival:20,erod:15,err:17,error:[2,13,15,18,20],especi:20,essenti:13,estim:[8,13,14,18,20],eta:12,etc:3,evalu:[1,9,11,14,15,20],even:20,eventu:[17,20],everi:[5,20],everyth:11,exact:[11,19],exact_lipschitz:[4,7],exampl:[1,5,7,11,13,17,18,20],except:[11,13,15,20],execut:[1,5,8,20],exist:[1,4,7,10,20],expect:[11,20],expens:20,explain:20,explan:18,explicitli:17,exposur:18,express:14,extend:[15,17,18,20],extendedsourc:[4,7,13,15,18,20],extened:18,extent:10,extra:20,extract:[12,13,15,17],extract_valu:12,factor:20,fail:8,faint:19,fainter:20,faintest:20,fals:[2,4,5,7,11,13,15,17,19,20],familiar:[13,17],far:20,fast:20,faster:[4,7,20],featur:20,feel:[18,20],few:20,fft:[3,9,10,13,18,20],fft_convolv:9,fft_resampl:9,field:[14,18,20],fig:[13,18,20],fig_height:13,fig_width:13,figsiz:[13,17,18,20],figur:[13,17,18,20],file:8,filer:20,fill:11,fill_valu:6,filter:[4,7,10,13,18,20],filter_weight:[6,20],filtercurv:10,find:[11,14,20],find_q:11,find_relevant_dim:11,first:[5,8,13,17,18,19,20],fit:[1,2,4,5,7,12,15,16,19],fit_pixel_cent:[19,20],fit_target_psf:[12,18],fix:[5,8,18,20],fix_morph:[4,5,7,18,20],fix_pixel_cent:[4,7],fix_s:[4,5,7],flag:[2,5,7,18],flat:15,flatten:[5,15,17],float64:11,floor:9,fluctuat:20,flux:[1,4,5,6,7,11,13,15,17,19,20],flux_at_edg:[1,4,7],flux_percentil:15,fluxand:20,folder:8,follow:[12,20],footprint:[15,19,20],forc:[4,7,11,15],form:[19,20],format:[11,13,17,18,20],formul:20,found:20,four:20,fourier:[4,7],frac:20,fraction:[4,7,9,17,20],frame:[4,7,9,10,11,14,17,20],fred3m:8,fred:[13,17,18,20],free:18,freedom:20,frequent:3,frobeniu:20,from:[4,7,9,10,11,12,13,14,15,17,18,19],full:[2,8,18],fulli:18,func:[11,12],fundament:[13,20],further:20,futur:[13,17,18,20],futurewarn:[13,18,20],gaia:18,galact:18,galaxi:[1,11,18,19,20],galxi:20,gase:20,gaussian:[12,13,20],gener:[1,5,9,12,13,15,18,20],geometr:20,geq:20,get:[4,5,9,10,11,13,15,17,18,20],get_best_fit_s:15,get_common_pad:9,get_default_filter_weight:20,get_flux:[4,5,7,20],get_loss:10,get_model:[4,5,7,10,13,17,18,20],get_model_imag:[10,17],get_pixel:10,get_pixel_s:[15,20],get_projection_slic:9,get_scen:10,get_scene_s:15,get_separable_kernel:9,get_true_imag:13,getoffset:11,getradialmonotonicop:11,getradialmonotonicweight:11,gist_stern:17,git:[4,7,8],github:[8,13,20],gitignor:8,give:20,given:[9,10,11,13,15,19,20],global:20,globalrm:17,goal:20,goe:18,gone:[4,7],good:[11,20],govern:20,grad_func:5,gradient:[4,5,7,8,11,20],greater:18,green:20,gri:20,grid:[4,7,9,10,14,19,20],grism:20,grizi:20,ground:20,group:[5,20],grow:[15,19],guarante:20,guess:[5,16,20],guid:[7,17,18],hack:17,had:20,half:10,handi:20,handl:[4,7,13,20],happen:20,happi:20,happili:20,has:[1,2,4,5,7,8,11,13,18,20],has_truth:[13,17],hasattr:20,have:[2,4,7,8,9,10,11,13,15,17,18,19,20],haven:20,header:[8,17],height:[1,5,6,15],her:20,here:[18,20],hierarch:[4,5,7],high:[4,7,9,10,14,17],higher:17,highest:11,highi:20,highr:17,his:20,histogram:19,histori:20,hold:[3,5],home:8,hope:18,hopefulli:18,host:[8,20],how:[13,17,18,20],howev:20,hsc:[11,13,17,20],hsc_cosmos_35:[13,20],hsc_norm:17,hst:17,hst_hdu:17,hst_img:17,html:8,http:[8,13],hubbl:17,hyper:[11,17],hyperplan:11,ibb:9,idea:20,ideal:20,idx:18,ignor:[8,20],illustr:20,imag:[1,4,5,6,7,9,10,11,12,14,15,17,18,20],imaginari:17,img1:9,img2:9,img:[6,9,13,15,17,20],img_hr:17,img_lr_rgb:17,img_rgb:[13,17,18,20],img_to_channel:[4,6,7,20],img_to_rgb:[6,13,17,18,20],immedi:19,implement:[4,7,13,17,20],impli:20,impos:20,imposs:20,improperli:20,improv:[4,7,17,20],imshow:[13,17,18,20],inabl:20,includ:[4,6,7,11,13,15,19,20],incorpor:20,incorrect:[5,20],increas:20,independ:20,index:[8,11,13,15,18,20],indic:[3,11,20],individu:[2,4,7,20],inexpens:3,inferno:[13,18,20],infinit:20,info:13,inform:[6,7,9,10,13,20],inherit:[4,5,7,15,20],init:18,init_combined_extended_sourc:15,init_extended_sourc:15,init_multicomponent_sourc:15,init_rgb:17,init_valu:12,initi:[4,5,7,12,15,16],initialz:[13,20],initiaz:20,inlin:[13,17,18,20],input:[1,4,7,9,13,20],insert:9,insid:[11,14,15,20],inspect:20,instal:[4,7,13],instanc:[5,13,20],instanci:10,instead:[4,7,13,18,20],instrument:17,insuffici:20,integ:1,integr:8,intens:[5,13,20],intensity_:13,interact:11,interfac:[4,7],intern:[2,4,7,13,20],interpol:[0,4,7,13,14,17,18,20],interpret:[13,18,20],intial:[15,16],introduc:[4,7],introduct:7,invers:14,involv:20,irregular:20,is_star:18,isn:9,iter:[2,4,7,11,13,14,17,18,20],its:[2,4,5,7,11,13,17,19,20],ixslic:9,iyslic:9,jet:20,jump:20,just:[1,4,7,10,13,19,20],kbarbari:13,keep:[4,7,17,20],kei:3,kept:11,kernel:[4,7,8,9,13,14,18,19,20],kernel_fft:17,kesi:20,keyword:[9,15],know:18,known:[11,15,20],kwarg:[9,11,20],l_2:20,l_func:5,lanczo:[4,7,9,19],lane:20,larg:[19,20],larger:[4,7,9,11,20],last:[17,20],later:[11,18,20],layer:15,lead:20,leaf:5,learn:20,least:[11,20],left:[1,9,20],len:[13,18,20],length:11,less:[11,17,20],let:18,level:[11,17,18,20],leverag:[7,20],librari:[7,8],like:[4,7,9,11,18,19,20],likelihood:[8,10,13,20],limit:20,linear:[6,9,13,14,17,20],linearmap:[13,17,18,20],linearpercentilenorm:6,linorm2d:14,lipschitz:[2,4,5,7,14,20],list:[2,4,5,7,9,12,13,15,17,20],live:[13,15],load:[7,8,16,18],local:[8,17],locat:[5,9,11,14,15,18,20],log:[13,17,19],logic:[4,7],longer:[4,7],look:20,lookup:3,loss:10,low:[9,13,14,15,17,20],lower:[4,7,9,20],lowr:17,lowresobserv:[4,7,10,17],lpha:12,lsst:[4,7,20],lupton:20,lupton_rgb:[13,17,18,20],m_k:20,machineri:13,mactch:14,made:[15,20],mai:[5,18,20],main:20,mainli:9,major:[11,20],make:[4,7,8,11,14,15,19,20],make_lupton_rgb:20,make_oper:14,makecatalog:17,mani:[7,11,20],manual:20,map:[4,6,7,13,14,17,18,20],mark:[13,20],markers:17,mask:[6,10,11,14],master:[8,20],mat:14,match:[0,4,7,10,13,14,15,17,18,20],match_patch:14,match_psf:14,mathbb:20,mathemat:[13,20],mathsf:20,matplotlib:[6,7,8,13,17,18,20],matric:11,matrix:[1,4,7,11,14,15,18,19,20],max:[11,12,13,17,18,20],max_it:[2,20],maximum:[2,13,14,19,20],mean:[2,5,10,11,13,15,17,18,20],meantim:18,measur:20,melchior:20,member:2,mention:20,merg:20,meshgrid:[13,20],met:20,meta:10,metadata:[4,7,10,20],method:[1,2,3,4,5,6,7,9,10,12,14,15,17,18,19,20],mew:[13,18,20],might:[4,5,7,8,11,13,18,20],min:[12,17,18,20],min_a:15,min_valu:[1,20],mingradi:11,minim:[8,13,20],minimallyconstrainedsourc:20,minimum:[1,11,13,15,17,18,20],minimum_valu:1,minor:20,mislead:20,mitchel_netravali:9,mix:[13,20],mixtur:18,model:[2,4,5,6,7,9,10,11,12,15,16],model_hr:17,model_lr:17,model_rgb:[13,20],modifi:20,modul:[4,7,9,10,20],moffat:[12,18],monoton:[1,3,4,7,8,11,13,15,17,19],monotonicsourc:20,monton:20,more:[4,6,7,11,13,17,18,20],morph:[5,15,17,18,19,20],morph_grad:20,morph_hr:17,morph_lr:17,morph_max:[19,20],morph_not_converg:[5,20],morpholgi:19,morpholog:[4,5,7,9,11,15,17,18,19,20],most:[4,7,18,20],mse:2,much:[17,20],multi:[4,6,7,9,15,16,20],multiband:[6,13,17,20],multicomponentsourc:[4,7,15,20],multidimension:[13,18,20],multipl:[2,4,5,7,15,20],multipli:20,must:[8,10,12,13,17,20],n_compon:[5,15],n_node:[5,15],naiv:20,name:[3,4,7,13,20],narrow:[15,20],nativ:17,natur:20,navig:8,nbsphinx:8,ncol:20,nearbi:19,nearest:[8,9,11,20],necessari:[9,10,17,20],need:[1,3,4,7,8,9,10,11,13,17,18,20],neg:[19,20],neighbor:[8,9,11,19,20],neither:[15,20],never:[18,20],new_sourc:[13,18,20],newbyteord:17,next:[8,17],nit:14,no_valid_pixel:5,node:[5,15],nois:[13,18,19,20],non:[13,18,19,20],none:[1,5,6,9,10,11,12,13,15,17,18,19,20],nor:[15,20],norm:[6,13,17,18,19,20],normal:[4,6,7,11,13,18,19],notat:[13,20],note:[13,20],notebook:8,notic:[18,20],now:[4,7,10,11,17,20],np1:17,np2:17,npz:[13,18,20],nrow:20,nth:20,number:[2,4,5,7,10,13,14,15,18,20],numpi:[5,6,8,10,11,13,15,17,18,20],numpydoc:8,nxn:11,object:[1,4,5,7,10,14,15,18,20],obs:17,obs_hdu:17,obs_hsc:17,obs_hst:17,obs_idx:[15,17],observ:[0,2,4,7,11,15,16],observed_fft:17,obtain:20,occasion:13,odd:[4,7],off:11,offer:15,offset:[11,18],often:[6,20],old:[4,7],onc:[5,11,13,18,20],one:[4,5,7,11,13,14,19,20],ones:[13,18,20],onli:[5,10,11,13,15,18,19,20],onto:[3,9,11,14],opaqu:11,open:[7,13,17],oper:[0,3,4,5,7,14,19,20],oppos:[4,7,13,20],optim:[4,7,8,20],option:[4,5,7,8,13,20],order:[11,13,17,18,19,20],org:13,organ:20,orient:[4,7,14],origin:[11,13],other:[5,10,11,13,19,20],otherwis:[11,20],our:[13,17,18,20],out:[13,20],outer:[15,20],output:20,outsid:[6,9,11,18],outskirt:20,over:[1,19,20],overal:[15,20],overlap:[14,20],overli:20,overload:20,overview:20,overwrit:[4,7,15],overwritten:[5,15],own:[5,13,18,20],packag:[4,7,11,13,17],pad:[4,7,9,10,13,18,19,20],page:[8,20],paramet:[1,2,4,5,6,7,9,10,11,12,14,15,17,19,20],paremet:[9,20],parikh:20,part:[17,20],partial:18,particular:20,partner:11,pass:[4,7,11,15,18,20],patch:[4,7,11,13,20],path:8,pathtoscarlet:8,pattern:20,peak:[4,7,11,18,20],peigen:8,penalti:20,per:20,percentil:[6,15],perfect:20,perform:[4,5,7,14,18,20],perhap:20,perimet:15,pesquet:20,photometr:20,photometri:18,photutil:20,pixel:[1,4,5,6,7,9,10,11,13,14,15,17,18,19,20],pixel_cent:[13,18,19,20],place:[4,7,9,19,20],plane:[11,14],pleas:20,plot:[6,8,13,17,18,20],plt:[13,17,18,20],pnorm:18,point:[7,15,16,20],point_sourc:20,pointsourc:[4,7,13,15,18,20],popul:20,portion:[1,11,13,17],posit:[1,4,7,9,10,11,15,18,19],positive_morph:[19,20],positive_s:[19,20],possibl:[4,7,13,18,20],potenti:[4,7,20],power:14,practic:20,precis:20,predict:20,preserv:[6,20],prevent:[10,11,18,19,20],previou:[11,19],prgb:18,primari:15,print:[13,17,18,20],prior:[4,5,7,8,9,18,20],probabl:11,problem:[11,20],procedur:18,proceedur:20,process:[8,17],produc:20,product:9,profil:[1,20],proj:11,proj_dist:11,project:[4,7,9,10,11,13,17,18,19,20],project_disk_s:11,project_disk_sed_mean:11,project_imag:9,prone:17,proper:20,properli:[4,7,20],properti:[1,20],provid:[2,4,7,20],prox_center_on:11,prox_con:11,prox_max_un:11,prox_monoton:[11,19],prox_plu:20,prox_sdss_symmetri:11,prox_sed_on:11,prox_soft_symmetri:11,prox_strict_monoton:11,prox_uncentered_symmetri:[11,19],proxim:[0,3,5,7,8,19,20],proximal_disk_s:11,proxmin:[8,11,20],psf:[0,3,4,7,8,10,14,16,17,20],psf_hr:14,psf_hsc:17,psf_hst:17,psf_lr:14,psf_match_hr:14,psf_match_lr:14,psf_unmatched_sim:18,pull:20,purpos:[1,20],push:[3,8,20],put:[5,11,20],pybind11:8,pyplot:[13,17,18,20],python:[4,7,8],quick:[7,18],quickli:[11,13],rac:12,radial:[11,20],rail:11,ran:[13,17],random:18,rang:[6,13,17,18,19,20],ratio:13,raycast:11,reach:[15,20],read:8,reader:20,real:[4,7,17,20],realist:18,realli:1,reason:20,recal:20,recalcul:[4,7],recommend:[15,20],reconstruct:20,record:11,rectangular:[1,4,7],red:[13,18,20],redder:11,reduc:[11,13,20],refer:20,refin:17,regardless:[11,18],region:[1,4,7,11,14,15,18,19,20],regular:17,reimplement:20,rel:[2,3,4,7,8,20],releas:[1,4,7,20],relev:11,reliabl:20,remain:11,remaind:20,remov:[4,7,15,20],renam:[4,7],render:11,replac:[4,7,18],repo:[8,20],repres:[2,11,13,20],represent:[18,20],reproject:[4,7,10,14],request:20,requir:[1,4,5,7,8,11,18,20],res:17,res_rgb:17,resampl:[0,4,7,9,10],rescal:20,resconv_op:17,reshap:17,residu:[13,17,18,20],residual_hr:17,residual_lr:17,residual_lr_rgb:17,residual_rgb:13,resiz:[1,4,7],resolut:[4,7,9,10,14,15,16,20],resolv:20,respect:[14,15,17,20],rest:11,restrict:20,restructur:[4,7],result:[1,6,7,9,10,11,14,16,20],revers:20,rgb:[4,6,7,13,17,18,20],right:[1,9,20],rightarrow:[13,20],rigor:[11,20],risk:20,robert:20,robust:[18,20],rough:2,roughli:[11,13,17,20],routin:15,row:[11,20],rresolut:14,run:[2,4,5,7,13,17,20],same:[4,5,7,9,10,11,13,14,17,18,19,20],sampl:[7,8,9,16,18,20],sample_lr:9,satisfi:20,scalar:10,scale:[6,10,13,14,17,18,20],scarlet:[0,4,13,17,18,20],scatter:20,scenc:[10,18],scene:[0,2,4,5,7,8,9,13,14,15,16],scheme:[12,20],scienc:[13,20],scipi:[8,11],scope:20,sdss:[11,13,20],search:20,section:8,sed:[4,5,7,11,15,17,19,20],sed_grad:20,sed_not_converg:[5,20],see:[4,6,7,9,11,15,17,18,19,20],seek:14,seismic:17,sel:17,select:[4,7,11,20],self:[5,20],sens:[11,20],sensit:20,sep:[13,17,20],separ:[4,5,7,9,20],seq:[13,18,20],sequenc:[13,18,20],sersic:20,set:[3,4,5,7,10,11,13,15,17,18,19,20],set_titl:[13,18,20],set_xlabel:20,set_ylabel:20,set_ylim:20,setup:[8,13,17,20],sever:[6,8,20],shallow:18,shape:[1,4,5,7,9,10,11,13,14,15,17,18,20],shape_hr:14,shape_lr:14,sharp:9,shift:[4,7,9,19,20],should:[1,4,6,7,8,10,11,12,17,18,19,20],show:[13,17,18,20],show_s:20,shown:20,side:[1,10,18],sigma1:12,sigma2:12,sigma:[11,12,20],signal:20,signific:[15,20],similar:[4,6,7,11,20],similarli:20,simpl:[17,20],simplifi:[4,7],simplist:20,simul:[13,17,18],sin:[13,17],sinc2d:9,sinc:[9,14,17,20],sinc_interp:9,singl:[5,10,11,12,13,15,17,20],singular:[4,7],sinh:[13,17,20],size:[4,7,9,11,13,14,15,17,20],skip:15,sky:[2,13,20],sky_coord:[10,15],slack:18,slice:[1,4,7,9,11,13,18,19,20],slightli:[15,20],slow:19,small:[4,7,18],smaller:[4,7,20],smallest:1,smax:[4,7],smooth:20,smoother:20,soft:11,soften:20,softwar:[6,20],solut:[11,20],solv:15,some:[2,4,7,8,10,11,19,20],soon:1,sort:11,sort_by_radiu:11,sourc:[0,1,2,3,4,5,6,7,9,10,11,12,14,16,19],sourceiniterror:[13,15,20],space:[4,7,9,11,17,19,20],spars:[11,18],sparse_l0:19,sparse_l1:19,sparsiti:19,spatial:10,special:[8,20],specif:20,specifi:[11,12,15],spectral:[10,20],spectrum:[13,20],speed:[4,7],sphinx:8,sphinx_rtd_them:8,spiral:20,spline:[4,7,9],sprime:17,spurriou:20,sqrt:20,squar:2,src:[13,17,18,20],stabl:20,stack:[4,7,13,20],standard:17,star:[18,20],start:20,stellar:[18,20],step:[4,7,11,15,20],step_morph:[5,15],step_s:[5,15,20],still:20,stop:20,store:3,str:[13,18,20],strategi:13,strength:[11,19],stretch:[13,17,18,20],strict:11,strip:[4,7],structur:[10,17],struggl:20,style:8,sub:[4,7,11],subclass:[4,7],submit:20,suboptim:20,subpixel:20,subplot:[17,20],subregion:11,subsequ:15,subset:20,substanti:20,subtract:11,subvers:[4,7],successfulli:20,suffici:20,suit:20,sum:[12,13,18,19,20],sum_:20,support:[4,7,10,13],suptitl:17,sure:[8,11,20],swap:17,symmeti:20,symmetr:[11,12,13,15,17,19,20],symmetri:[1,4,7,11,15],sys:18,system:[10,18],take:[4,7,9,11,12,17,19,20],taken:[17,18],target:[4,7,8,10,11,12,18,20],target_fft:17,target_psf:[10,12,13,18,20],target_rgb:18,tbd:10,teh:17,telescop:[17,20],tell:17,templat:[13,20],tend:20,tensor:[1,5,9,10],term:20,terminolog:20,test:[2,4,7,18,20],test_resampl:17,text:[13,18,20],textrm:20,than:[2,4,7,9,11,18,20],thei:[6,8,20],them:[4,7,8,11,15,20],theme:8,thi:[1,2,4,5,6,7,8,9,10,11,13,14,15,17,18,19,20],thing:[4,7],think:20,those:[11,13,20],though:[4,7],thought:20,three:[17,18,20],thresh:[1,11,15,19],threshold:[1,5,13,15,19],throughout:20,thu:[4,7],tight_layout:20,tightli:20,time:[3,5,15,17,18,19,20],times60:1,tini:11,titl:[17,18],togeth:[5,20],too:20,tool:[10,20],top:[1,9],tose:20,total:[5,18,20],toward:20,track:[4,7,20],tradeoff:20,tradit:[13,20],transform:[3,4,7,10],translat:[9,13,19],travers:5,travi:[4,7],treat:20,tree:[5,13,15,20],trigger:1,trim:[1,4,7,9,17,20],truli:20,truncat:[4,6,7],tupl:[1,5,9,10,11,13,14,15,18,19,20],turn:15,tutori:[4,7,8,13,17,20],tweak:20,two:[9,11,12,14,17,20],type:[8,10,18,19,20],typic:20,typicali:20,uint8:6,unbound:1,uncentered_oper:[4,7,11],unchang:11,uncommit:8,under:20,underli:20,understand:20,undetect:20,unecessari:11,unfortun:18,uniqu:18,unit:5,uniti:[11,19,20],unknown:20,unless:[8,20],unlik:[11,17],unpack:10,unphys:19,until:15,updat:[0,1,4,5,7,11,15,18],update_histori:20,upsampl:17,use:[1,2,3,4,5,6,7,8,9,11,12,13,15,17,18,19,20],use_fft:[4,7],use_nearest:[11,19,20],use_prox:19,use_relevant_dim:11,use_soft:11,used:[1,2,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,20],useful:[6,9,13,15,17,20],usenearest:11,user:[4,7,11,13,17,18],uses:[1,9,11,13,20],using:[4,7,8,11,13,17,18,20],usual:[8,18,20],util:11,valu:[1,3,4,6,7,9,11,12,13,15,17,18,19,20],valueerror:12,vari:11,vascin:19,vector:[11,14,20],veri:[19,20],version:[1,4,7,8,11,17,18,20],vertic:15,view:[5,7,15,20],violat:20,virtual:20,visual:[0,4,7,8,13,17,18,20],vmax:17,vmin:17,wai:[11,13,18,20],wall:18,want:[8,11,20],warrant:1,watch:20,wavelength:11,wcs:[10,14,17],wcs_hr:14,wcs_hsc:17,wcs_hst:17,wcs_lr:14,wcs_pix2world:17,wcs_world2pix:17,week:20,weight:[10,11,15,20],well:[9,13,14,17,19,20],were:[4,5,7],when:[2,4,7,8,10,11,13,18,19,20],where:[9,11,14,17,18,20],wherea:[1,18],whether:[1,2,5,11,15,19,20],which:[1,2,3,4,7,8,10,11,13,14,17,18,20],white:20,why:20,wide:[19,20],width:[1,5,6,10,11,15],window:[9,19,20],wing:20,wise:20,wish:8,within:[2,11,19],without:20,word:13,work:[11,19,20],world:10,worri:20,worst:20,would:[4,7,20],wound:20,write:20,wrong:[4,7],www:13,x_window:9,xslice:9,y_window:9,yet:[5,13,20],yield:11,you:[8,13,17,18,20],your:[8,13,18,20],yslice:9,yx0:9,zero:[9,10,11,17,20]},titles:["API Documentation","Bounding Boxes (<code class=\"docutils literal notranslate\"><span class=\"pre\">scarlet.bbox</span></code>)","Blend (<code class=\"docutils literal notranslate\"><span class=\"pre\">scarlet.blend</span></code>)","Cache (<code class=\"docutils literal notranslate\"><span class=\"pre\">scarlet.cache</span></code>)","0.6 (unreleased)","Components (<code class=\"docutils literal notranslate\"><span class=\"pre\">scarlet.component</span></code>)","Visualization (<code class=\"docutils literal notranslate\"><span class=\"pre\">scarlet.display</span></code>)","<em>SCARLET</em>","Installation","Interpolation (<code class=\"docutils literal notranslate\"><span class=\"pre\">scarlet.interpolation</span></code>)","Observations and Scenes (<code class=\"docutils literal notranslate\"><span class=\"pre\">scarlet.observation</span></code>)","Proximal Operators (<code class=\"docutils literal notranslate\"><span class=\"pre\">scarlet.operator</span></code>)","PSF matching (<code class=\"docutils literal notranslate\"><span class=\"pre\">scarlet.psf</span></code>)","Quick Start Guide","Resampling (<code class=\"docutils literal notranslate\"><span class=\"pre\">scarlet.resampling</span></code>)","Sources (<code class=\"docutils literal notranslate\"><span class=\"pre\">scarlet.source</span></code>)","Tutorials","Multi-resolution Deblending","Point Source Tutorial","Update (<code class=\"docutils literal notranslate\"><span class=\"pre\">scarlet.update</span></code>)","User Guide"],titleterms:{"import":20,"new":[4,7,20],addit:[4,7],api:[0,1,2,3,4,5,6,7,9,10,11,12,14,15,19],basic:20,bbox:1,blend:[2,17,18,20],blendflag:20,bound:1,box:1,bug:[4,7],build:8,cach:3,catalog:13,center:20,chang:[4,7],check:20,compon:[5,20],concept:20,constraint:20,construct:20,converg:20,creat:[13,17,18],cube:13,data:[13,17,20],deblend:17,develop:8,displai:[6,13,17,18,20],doc:8,document:0,extract:20,featur:[4,7],fit:[13,17,18,20],fix:[4,7],flag:20,from:[8,20],full:[13,17],gener:[4,7],get:7,guess:17,guid:[13,20],hierarch:20,imag:13,initi:[13,17,18,20],instal:8,interpol:9,intial:17,introduct:20,load:[13,17,20],log:7,match:12,model:[13,17,18,20],monoton:20,multi:17,normal:20,observ:[10,13,17,18,20],onli:8,oper:11,other:[4,7],packag:20,point:18,posit:20,proxim:11,psf:[12,13,18],quick:13,raw:13,refer:[1,2,3,5,6,9,10,11,12,14,15,19],resampl:14,resolut:17,restart:20,result:[13,18],sampl:[13,17],scarlet:[1,2,3,5,6,7,8,9,10,11,12,14,15,19],scene:[10,17,18,20],sourc:[8,13,15,17,18,20],start:[7,13],structur:20,symmetri:20,target:13,tutori:[16,18],unreleas:[4,7],updat:[8,19,20],user:20,view:[13,17],visual:6}})
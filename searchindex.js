Search.setIndex({docnames:["0-quickstart","1-concepts","api/scarlet","api/scarlet.bbox","api/scarlet.blend","api/scarlet.cache","api/scarlet.component","api/scarlet.constraint","api/scarlet.display","api/scarlet.fft","api/scarlet.frame","api/scarlet.interpolation","api/scarlet.measure","api/scarlet.observation","api/scarlet.operator","api/scarlet.parameter","api/scarlet.prior","api/scarlet.psf","api/scarlet.resampling","api/scarlet.source","changes","index","install","tutorials","tutorials/display","tutorials/multiresolution","tutorials/point_source"],envversion:{"sphinx.domains.c":1,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":1,"sphinx.domains.javascript":1,"sphinx.domains.math":2,"sphinx.domains.python":1,"sphinx.domains.rst":1,"sphinx.domains.std":1,"sphinx.ext.intersphinx":1,"sphinx.ext.viewcode":1,nbsphinx:1,sphinx:56},filenames:["0-quickstart.ipynb","1-concepts.ipynb","api/scarlet.rst","api/scarlet.bbox.rst","api/scarlet.blend.rst","api/scarlet.cache.rst","api/scarlet.component.rst","api/scarlet.constraint.rst","api/scarlet.display.rst","api/scarlet.fft.rst","api/scarlet.frame.rst","api/scarlet.interpolation.rst","api/scarlet.measure.rst","api/scarlet.observation.rst","api/scarlet.operator.rst","api/scarlet.parameter.rst","api/scarlet.prior.rst","api/scarlet.psf.rst","api/scarlet.resampling.rst","api/scarlet.source.rst","changes.rst","index.rst","install.rst","tutorials.rst","tutorials/display.ipynb","tutorials/multiresolution.ipynb","tutorials/point_source.ipynb"],objects:{"scarlet.bbox":{Box:[3,1,1,""]},"scarlet.bbox.Box":{D:[3,2,1,""],contains:[3,2,1,""],extract_from:[3,2,1,""],from_bounds:[3,2,1,""],from_data:[3,2,1,""],from_image:[3,2,1,""],insert_into:[3,2,1,""],slices_for:[3,2,1,""],start:[3,2,1,""],stop:[3,2,1,""]},"scarlet.blend":{Blend:[4,1,1,""]},"scarlet.blend.Blend":{fit:[4,2,1,""]},"scarlet.cache":{Cache:[5,1,1,""]},"scarlet.cache.Cache":{check:[5,2,1,""],set:[5,2,1,""]},"scarlet.component":{Component:[6,1,1,""],ComponentTree:[6,1,1,""],CubeComponent:[6,1,1,""],FactorizedComponent:[6,1,1,""],FunctionComponent:[6,1,1,""]},"scarlet.component.Component":{check_parameters:[6,2,1,""],coord:[6,2,1,""],get_model:[6,2,1,""],parameters:[6,2,1,""],set_frame:[6,2,1,""],shape:[6,2,1,""]},"scarlet.component.ComponentTree":{K:[6,2,1,""],check_parameters:[6,2,1,""],components:[6,2,1,""],coord:[6,2,1,""],frame:[6,2,1,""],get_model:[6,2,1,""],n_components:[6,2,1,""],n_sources:[6,2,1,""],parameters:[6,2,1,""],set_frame:[6,2,1,""],sources:[6,2,1,""]},"scarlet.component.CubeComponent":{cube:[6,2,1,""],get_model:[6,2,1,""]},"scarlet.component.FactorizedComponent":{get_model:[6,2,1,""],morph:[6,2,1,""],sed:[6,2,1,""],shift:[6,2,1,""]},"scarlet.component.FunctionComponent":{get_model:[6,2,1,""],morph:[6,2,1,""]},"scarlet.constraint":{AllOnConstraint:[7,1,1,""],CenterOnConstraint:[7,1,1,""],Constraint:[7,1,1,""],ConstraintChain:[7,1,1,""],L0Constraint:[7,1,1,""],L1Constraint:[7,1,1,""],MonotonicityConstraint:[7,1,1,""],NormalizationConstraint:[7,1,1,""],PositivityConstraint:[7,1,1,""],SymmetryConstraint:[7,1,1,""],ThresholdConstraint:[7,1,1,""]},"scarlet.constraint.ThresholdConstraint":{threshold:[7,2,1,""]},"scarlet.display":{AsinhPercentileNorm:[8,1,1,""],LinearPercentileNorm:[8,1,1,""],channels_to_rgb:[8,3,1,""],img_to_3channel:[8,3,1,""],img_to_rgb:[8,3,1,""],show_scene:[8,3,1,""],show_sources:[8,3,1,""]},"scarlet.fft":{Fourier:[9,1,1,""],convolve:[9,3,1,""],match_psfs:[9,3,1,""]},"scarlet.fft.Fourier":{fft:[9,2,1,""],from_fft:[9,2,1,""],image:[9,2,1,""],shape:[9,2,1,""]},"scarlet.frame":{Frame:[10,1,1,""]},"scarlet.frame.Frame":{C:[10,2,1,""],Nx:[10,2,1,""],Ny:[10,2,1,""],get_pixel:[10,2,1,""],get_sky_coord:[10,2,1,""],psf:[10,2,1,""]},"scarlet.interpolation":{apply_2D_trapezoid_rule:[11,3,1,""],bilinear:[11,3,1,""],catmull_rom:[11,3,1,""],common_projections:[11,3,1,""],cubic_spline:[11,3,1,""],fft_convolve:[11,3,1,""],fft_resample:[11,3,1,""],get_common_padding:[11,3,1,""],get_projection_slices:[11,3,1,""],get_separable_kernel:[11,3,1,""],lanczos:[11,3,1,""],mitchel_netravali:[11,3,1,""],mk_shifter:[11,3,1,""],project_image:[11,3,1,""],quintic_spline:[11,3,1,""],sinc2D:[11,3,1,""],sinc_interp:[11,3,1,""],subsample_function:[11,3,1,""]},"scarlet.measure":{centroid:[12,3,1,""],flux:[12,3,1,""],get_model:[12,3,1,""],max_pixel:[12,3,1,""]},"scarlet.observation":{LowResObservation:[13,1,1,""],Observation:[13,1,1,""]},"scarlet.observation.LowResObservation":{build_diffkernel:[13,2,1,""],get_loss:[13,2,1,""],match:[13,2,1,""],match_psfs:[13,2,1,""],render:[13,2,1,""],sinc_shift:[13,2,1,""]},"scarlet.observation.Observation":{get_loss:[13,2,1,""],match:[13,2,1,""],render:[13,2,1,""]},"scarlet.operator":{diagonalizeArray:[14,3,1,""],diagonalsToSparse:[14,3,1,""],find_Q:[14,3,1,""],find_relevant_dim:[14,3,1,""],getOffsets:[14,3,1,""],getRadialMonotonicOp:[14,3,1,""],getRadialMonotonicWeights:[14,3,1,""],proj:[14,3,1,""],proj_dist:[14,3,1,""],project_disk_sed:[14,3,1,""],project_disk_sed_mean:[14,3,1,""],prox_cone:[14,3,1,""],prox_kspace_symmetry:[14,3,1,""],prox_sdss_symmetry:[14,3,1,""],prox_soft_symmetry:[14,3,1,""],prox_strict_monotonic:[14,3,1,""],prox_uncentered_symmetry:[14,3,1,""],proximal_disk_sed:[14,3,1,""],sort_by_radius:[14,3,1,""],uncentered_operator:[14,3,1,""],use_relevant_dim:[14,3,1,""]},"scarlet.parameter":{Parameter:[15,1,1,""],relative_step:[15,3,1,""]},"scarlet.prior":{Prior:[16,1,1,""]},"scarlet.prior.Prior":{grad:[16,2,1,""]},"scarlet.psf":{PSF:[17,1,1,""],gaussian:[17,3,1,""],moffat:[17,3,1,""]},"scarlet.psf.PSF":{image:[17,2,1,""],normalize:[17,2,1,""],update_dtype:[17,2,1,""]},"scarlet.resampling":{match_patches:[18,3,1,""]},"scarlet.source":{ExtendedSource:[19,1,1,""],MultiComponentSource:[19,1,1,""],PointSource:[19,1,1,""],RandomSource:[19,1,1,""],build_detection_coadd:[19,3,1,""],get_best_fit_seds:[19,3,1,""],get_pixel_sed:[19,3,1,""],get_psf_sed:[19,3,1,""],init_extended_source:[19,3,1,""],init_multicomponent_source:[19,3,1,""],trim_morphology:[19,3,1,""]},"scarlet.source.ExtendedSource":{center:[19,2,1,""]},"scarlet.source.MultiComponentSource":{bbox:[19,2,1,""],center:[19,2,1,""],shift:[19,2,1,""]},scarlet:{bbox:[3,0,0,"-"],blend:[4,0,0,"-"],cache:[5,0,0,"-"],component:[6,0,0,"-"],constraint:[7,0,0,"-"],display:[8,0,0,"-"],fft:[9,0,0,"-"],frame:[10,0,0,"-"],interpolation:[11,0,0,"-"],measure:[12,0,0,"-"],observation:[13,0,0,"-"],operator:[14,0,0,"-"],parameter:[15,0,0,"-"],prior:[16,0,0,"-"],psf:[17,0,0,"-"],resampling:[18,0,0,"-"],source:[19,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:function"},terms:{"0x7f2291de4f28":11,"0x7f2291de5e18":14,"0x7f3e19b7c4e0":1,"0x7f3e288e7bf8":1,"0x7f3e43574160":1,"0x7f6d920aba20":0,"1st":11,"2nd":11,"2x2":11,"2xm":11,"2xn":11,"4750118475607095e":25,"8xn":14,"abstract":[6,16],"break":[20,21],"byte":[15,25],"case":[0,1,14,19,20,21,22,24],"class":[0,1,3,4,5,6,7,8,9,10,11,13,14,15,16,17,19,20,21,25],"default":[0,1,8,20,21,22,24,25],"final":[1,25],"float":[1,3,4,8,11,13,14,15,17,19],"function":[0,1,4,6,7,8,11,14,17,20,21,24],"import":[0,1,20,21,24,25,26],"int":[4,6,7,8,9,11,13,14],"long":1,"new":[0,3,9,11,14,24],"return":[1,3,6,8,9,10,11,13,14,17,18,19,20,21,25],"short":1,"static":[3,5,9],"super":1,"switch":24,"throw":14,"true":[0,1,6,8,14,17,18,19,24,25,26],"try":0,"while":[1,6,11,14,20,21,24,26],Axes:9,But:1,For:[0,1,7,8,14,19,24,25],Into:1,RMS:[19,25],That:[0,1,20,21],The:[0,1,3,4,6,7,8,9,11,13,14,15,19,20,21,22,24,25,26],Then:[22,25],There:1,These:1,Use:[7,11,26],Uses:[6,7,14,19],Using:[1,14],WCS:[0,1,10,13,18,25],With:[0,1],__call__:1,__class__:1,__dict__:1,__getitem__:6,__init__:[1,6],__name__:1,_build:22,abc:[6,16],abl:1,about:[0,1,3,7,11,15,26],abov:[0,1,3,7,22,24],absolut:1,acceler:[20,21],accept:[1,20,21,22],access:[1,20,21],accord:[3,7],account:0,accur:[0,14],across:[1,8,26],act:[6,7,14],actual:[0,1],adam:1,adaprox:[1,4,20,21],adapt:[1,20,21],add:[0,1,7,24],add_subplot:0,added:[1,20,21],adding:7,addit:[0,1,9,13,26],additon:1,adequ:13,adjust:[1,23],adopt:6,advantag:1,advis:14,affect:25,after:[14,15,26],against:[1,14],agnost:3,alg_kwarg:4,algorithm:[1,4,14,20,21,22],align:[1,14],all:[0,1,5,6,7,8,14,17,19,20,21,24,25,26],allonconstraint:7,allow:[0,1,7,8,14,20,21,22,24],alon:22,along:[9,13,20,21],alpha:17,alreadi:[0,22],also:[0,1,24,25],altern:[1,7],alwai:[14,20,21],ambigu:1,amount:[1,11,14],amplitud:[0,1,26],analysi:1,analyt:1,analyz:12,angl:[11,13,14],ani:[0,1,8,9,13,20,21,24,25],anoth:1,ansatz:1,anywher:[20,21],api:[0,1],appear:[1,14,24],append:[0,26],appli:[8,13,14,20,21,25],apply_2d_trapezoid_rul:11,apply_trapezoid_rul:11,approach:0,appropri:[14,26],approxim:[0,19,20,21],arbitrari:[20,21],arcsinh:24,argument:[1,6,8,10,11,13,20,21,25],aris:1,arithmeticerror:6,around:[14,24],arr:14,arrai:[0,1,3,4,6,8,9,11,13,14,15,17,18,19,25,26],arrang:0,array_lik:8,artifact:[9,13],asinh:[24,25,26],asinhmap:[0,8,24,25,26],asinhpercentilenorm:8,aspect:[0,1],assess:24,associ:10,assum:[0,1,8,11,19,20,21,24],assumpt:1,astrometr:1,astronom:21,astrophys:1,astropi:[0,1,8,20,21,22,24,25],attach:[6,20,21],attempt:[22,26],attribut:[1,3,4,6,9,10,13,15,17,19],autograd:[6,20,21,22],automat:22,auxiliari:6,avail:[0,1,6,22],averag:[1,14,25],avoid:[1,14],awai:[14,20,21],axes:[9,13,20,21,25],axi:[13,24,25],back:[3,14],background:[1,19,24,25,26],band:[0,1,6,8,13,14,17,20,21,24,25,26],bare:0,base:[0,1,3,4,5,6,7,8,9,10,11,13,14,15,16,17,19,20,21],basic:1,bbox:[2,6,7,10,17,19,20,21],becaus:[0,1,6,14,20,21,25],been:[1,9,20,21],befor:[0,9,11,22],beforehand:1,behavior:[1,20,21],being:[1,14],below:[1,7],best:[0,19,20,21],beta:[17,24],better:[1,14,20,21,25,26],between:[0,1,8,9,11,13,14,18,19,25,26],bg_cutoff:19,bg_rm:[19,20,21,25],bg_rms_hsc:25,bg_rms_hst:25,bilinear:[11,20,21],binari:[8,20,21],bit:[14,25,26],bkg:25,black:24,blend:[0,2,6,13,20,21,23],blend_:0,blendflag:[20,21],bloat:0,blue:[0,26],bluer:14,blur:0,bookkeep:[20,21],bool:[8,14,15,17],both:[0,1,14,18,25,26],bottom:[11,25],bound:[1,3,6,11,17,20,21,25],boundingbox:[20,21],box:[0,1,3,6,10,11,14,17,20,21,25],boyd:1,bright:24,bring:[1,24],broad:[1,24],broadband:1,buffer:15,bug:25,build:[1,6,11,13,14,19,20,21],build_detection_coadd:19,build_diffkernel:13,build_ext:22,built:0,bulg:14,bulge_s:14,byteswap:25,cach:[2,21],calcul:[0,9,11,14,19,20,21,22,25],call:[0,1,9,20,21],callabl:17,cam:25,can:[0,1,5,6,9,11,14,17,20,21,24,25,26],cannot:[1,14],captur:1,carri:24,catalog:[0,24,25,26],catalog_hsc:25,catalog_hst:25,cataog:0,catmull_rom:11,caus:[20,21],caution:6,celesti:24,center:[0,1,6,7,11,13,14,17,19,20,21,24,26],centeronconstraint:7,central:[1,14],centroid:[0,12,20,21],chain:7,chang:[0,1],channel:[0,1,3,6,8,10,12,13,19,20,21,24,25,26],channel_map:[8,24],channels_hsc:25,channels_hst:25,channels_to_rgb:[8,24],characterist:[6,10,13],check:[5,6],check_paramet:6,choic:24,choos:[0,1],chosen:[5,24],circular:17,clang:22,clarifi:[20,21],cleanli:24,clearli:[0,26],clone:22,close:[0,1,24],closer:[1,14],closest:[1,7],cmap:[0,24,25,26],coadd:19,code:[0,3,20,21,22],collect:[0,1,4,6,20,21],color:[0,1,8,14,20,21,24,25,26],colorbar:[1,25],colormap:[0,24,25,26],column:[13,14],com:22,combett:1,combin:[0,1,8,14,22,24],combinedextendedsourc:[20,21],come:1,command:[0,22],commandlinetool:22,commit:[20,21],common:[1,11],common_project:11,compar:[14,24,25],comparison:[0,14],compat:1,compil:22,complain:1,complet:[1,14,20,21],complex:[1,5,6,9],complic:[1,20,21],compoent:0,compon:[0,2,4,7,8,11,12,14,19,20,21,24],componentlist:[20,21],componenttre:[1,4,6,12,19,20,21],compress:8,comput:[0,1,13,16],concept:[0,7,21],conda:22,cone:14,config:[20,21],configur:13,confin:[20,21],confirm:26,confus:[20,21],connect:1,consequ:7,consid:8,consist:[0,24,25],constant:[20,21],constrain:[0,1,7,14,20,21],constraint:[0,2,15,20,21,26],constraintchain:[1,7],construct:[0,1,6,14],consum:15,contain:[1,3,7,9,11,14,19,20,21,25],content:[1,5],contrast:1,conveni:[1,24],convent:[0,3,5],converg:[0,1,4,14],convert:[8,14,24,25,26],convex:7,convinc:26,convolut:[0,1,9,11,13,20,21],convolv:[9,11,13],convov:11,coord:[6,14,19],coord_hr:11,coord_lr:11,coordhr_hr:18,coordin:[0,1,3,6,10,11,13,14,17,18,19,25],coordlr_over_hr:18,coordlr_over_lr:18,copi:22,core:[0,17,21,26],corner:[3,11],correct:[1,19,20,21,25,26],correctli:[20,21],correl:1,correspond:8,cos:13,cost:1,could:[0,1,14],counter:[1,15],cover:[1,3,6],coverag:[0,13,18],cpu:[0,25,26],creat:[1,8,9,11,14,18,19,20,21,23],crictic:0,criterion:7,critic:24,crowd:26,crval:25,ctype:15,cube:[1,6,8,10,13,20,21,24,26],cubecompon:[1,6],cubic:[11,20,21],cubic_splin:11,cubix:11,curiou:[1,26],current:[1,20,21,22],custom:[1,14],cut_hsc:25,cut_hst:25,cutoff:[7,20,21],data:[1,3,4,6,10,13,14,15,17,21,23,24,26],data_hsc:25,data_hst:25,dataset:[13,18],deal:0,debend:[20,21],deblend:[1,20,21],dec:25,decid:1,declar:1,decreas:[1,7],deeper:[1,26],def:[1,25],defin:[1,3,14,19,20,21,23,24,25],degener:1,degeneraci:[1,20,21],degre:1,demand:1,demonstr:[1,24,26],denot:3,depend:22,deploy:22,deprec:22,depth:[0,1],deriv:[20,21],descent:1,describ:[0,1,3,14,17],descript:[0,1,7,11,19,24],desir:[0,1,9],detail:[1,6,8,11,14,15],detect:[0,19,25],determin:[1,7,12,20,21,24],deviat:[1,17],diag:14,diagon:14,diagonalizearrai:14,diagonalstospars:14,dict:[4,6,11],dictionari:9,didx:14,diff_psf:13,differ:[0,1,6,9,11,13,14,18,19,20,21,25,26],differenti:[1,7,13],diffus:7,dimens:[0,3,8,9,11,14,15],dimension:[3,14],direct:[1,8,10,11],directli:0,directori:22,discard:14,discontinu:14,discuss:1,disk:14,disk_s:14,displai:[2,13,20,21,23],distanc:14,distinguish:1,distribut:[1,15,16,19],diverg:[20,21],dnx:11,dny:11,doabl:1,doc:[20,21],docstr:22,document:[0,1],doe:[1,7,8,14,20,21,22,25],doesn:[4,20,21],domain:11,domin:0,don:[0,1,22,24,25,26],done:[0,20,21,25],download:22,draw:0,drop:[20,21],dtype:[8,10,11,14,15,17],due:[20,21],dump:0,dure:[1,14,15,22,26],e_rel:[0,1,4],each:[0,1,4,6,8,9,10,13,14,15,19,20,21,24,25],earli:0,earlier:1,easier:[1,20,21],easiest:22,easili:[0,1],edg:[11,20,21],effect:[1,14,26],eigen:22,either:1,element:[0,1,6,7,10,14,15,20,21],elif:0,els:[0,1,25,26],emphasi:1,emploi:1,employ:1,enabl:[0,1],encod:[1,7,16],encount:6,end:0,endian:25,enforc:[1,14],enough:[1,26],ensur:[1,7,14],enter:1,entir:[0,1,8,20,21,25],enumer:[0,1,24,25],epoch:21,equal:[1,24],erod:19,err:25,error:[1,4,15,22],especi:[1,26],estim:[1,15,26],eta:17,etc:[0,1,5],euclidean:1,evalu:[8,11,14,17,19],even:[1,8,9,24],eventu:1,everi:[0,1,3,7,8,12,17,24],everyth:[0,14],exact:14,exact_lipschitz:[20,21],exactli:0,exampl:[0,1,9,14,25],except:14,exclud:15,execut:1,exist:[13,20,21],expect:[0,1,7,11,14,25],expens:1,explain:1,exploit:1,express:6,extend:[0,1,19,26],extendedsourc:[0,1,14,19,20,21,25,26],extent:1,extra:[9,22],extract:[3,19,25],extract_from:3,f814w:25,fact:[1,14],factor:[1,6,15],factorizedcompon:[1,6,19],factorizedmodel:1,faint:[7,24],fals:[0,1,7,8,11,14,19,20,21,25],far:1,fast:[1,8,9,20,21],faster:[20,21],feasibl:7,featur:[0,24],few:1,fft:[2,11,13,20,21],fft_convolv:11,fft_resampl:11,fft_shape:9,fidel:13,field:[1,18,19,26],fig:[0,24],figsiz:[0,8,24,25],figur:[0,8,25],file:[0,22],fill:14,fill_valu:[1,8],filter:[0,1,13,20,21,23,26],filter_curv:[20,21],find:[1,7,14,18],find_q:14,find_relevant_dim:14,finit:6,first:[0,6,15,24,25,26],fit:[1,4,7,8,19,20,21,23],fitter:[0,1],fix:[1,6,15,19],fix_morph:[20,21],fix_pixel_cent:[20,21],fix_s:[20,21],flag:15,flat:[15,19],flatten:[6,19],float32:10,float64:[11,14],floor:11,fluent:0,flux:[1,7,12,14,20,21,24,25],flux_at_edg:[20,21],flux_percentil:19,folder:22,follow:7,footprint:[7,19,20,21],forbidden:1,forc:[14,20,21],form:1,format:[0,1,14,24,25,26],found:22,fourier:[9,11,13,14,20,21],fparam:6,frac:[1,24],fraction:[1,11,14,20,21,25],frame:[2,3,6,11,13,14,18,19,20,21,23,24],free:1,freedom:[0,1],frobeniu:1,from:[0,1,3,7,9,10,11,13,14,15,19,20,21,22,24,25,26],from_bound:3,from_data:3,from_fft:9,from_imag:3,frome:25,full:[1,3,6,22,24],fulli:0,func:[6,14],functioncompon:[1,6,19],functool:[0,1,26],fundament:1,further:[1,20,21],futur:25,gaia:26,galaxi:[0,1,7,14,25,26],gase:1,gaussian:[0,1,17,26],gener:[0,1,3,6,9,11,14,17,19],get:[0,6,8,10,11,14,19,20,22,25],get_best_fit_s:[1,19],get_common_pad:11,get_flux:[20,21],get_loss:13,get_model:[0,6,12,20,21,25,26],get_pixel:10,get_pixel_s:19,get_projection_slic:11,get_psf_s:19,get_separable_kernel:11,get_sky_coord:10,getoffset:14,getradialmonotonicop:14,getradialmonotonicweight:14,getsourc:1,gist_stern:25,git:[20,21,22],github:22,give:24,given:[0,1,3,7,9,11,13,14,17,19],global:3,globalrm:25,goal:1,goe:24,going:0,gone:[20,21],good:[0,1,14,24],grad:[1,16],gradient:[1,7,14,15,16,20,21,22],grid:[11,13,20,21],grism:1,ground:0,grow:7,guess:[1,23,26],guid:[1,21],guidelin:1,hack:25,half:[1,13],handl:[20,21],happili:0,hard:1,has:[0,1,9,20,21,22],has_truth:25,hashabl:10,have:[0,1,6,7,11,13,14,15,20,21,22,24,25,26],header:[0,22,25],heheh:13,height:[3,6,8,10,19],held:[1,15],help:[1,20,21],here:[0,1,6,8,24],hierarch:[1,6,20,21],high:[11,13,18,20,21,25],higher:[0,25],highest:14,highi:1,highr:25,hing:1,histogram:7,hold:[0,1,5,6,15,22,25],home:22,homogen:1,horizont:17,host:1,how:[0,1,7,9,25,26],howev:[0,1,14],hsc:[1,14,25],hsc_cosmos_35:[0,1,24],hsc_norm:25,hst:25,hst_hdu:25,hst_img:25,hst_norm:25,html:22,http:22,hubbl:25,human:[8,24],hyper:[1,6,14,25],hyperplan:14,hyperspectr:[0,1,24],ibb:11,ident:[0,19],identifi:[3,10,15,20,21],idx:26,im_or_shap:3,imag:[1,3,6,8,9,10,11,13,14,15,17,19,20,21,24,25,26],image1:9,image2:9,image_fft:9,image_model:13,image_shap:9,imaginari:[14,15],img1:11,img2:11,img:[8,11,13,19,25],img_hr:25,img_lr_rgb:25,img_rgb:[0,24,25,26],img_to_3channel:8,img_to_channel:[20,21],img_to_rgb:[0,8,24,25,26],impact:24,implement:[0,1,11,20,21,24,25],impli:1,implicitli:[1,24],imposs:1,improp:1,improv:[0,1,20,21,25],imshow:[0,1,24,25,26],includ:[1,13,19,20,21],incorrect:1,incorrectli:1,independ:1,index:[8,14,17,22,26],indic:14,individu:[0,1,8,20,21,24],inevit:1,infer:1,inferno:[0,24,25,26],inform:[0,1,6,9,11,15,20,21],inherit:[20,21],init:26,init_extended_sourc:19,init_multicomponent_sourc:19,init_rgb:25,init_rgb_lr:25,initi:[1,3,6,14,19,20,21,23,26],inlin:[0,1,24,25,26],input:[1,3,11,20,21,24],insert:[1,3,11],insert_into:3,insid:[14,18,19],insight:1,inspect:[0,1,8,20,21,24],instal:[20,21],instanc:[0,1,9,13,20,21],instead:[0,1,20,21,22],instruct:0,instrument:[1,21,25],integr:[0,11,17,22],intens:[1,8,24],interact:[1,14,15,21],interest:1,interfac:[1,20,21],intern:[0,1,4,14,20,21],interpol:[0,2,13,20,21,24,25,26],intersect:[1,18],intial:[1,19],introduc:[1,20,21],introduct:0,invers:[1,3],involv:[20,21],irfft:9,is_star:26,isn:11,isol:24,isrot:18,issu:[1,20,21],items:15,iter:[0,1,4,14,15,20,21,25,26],its:[0,1,3,6,7,8,9,10,14,20,21,24,25],itself:[1,24],ixslic:11,iyslic:11,just:[1,10,20,21,24],kale:15,keep:[14,20,21,25],kei:[1,5,24],kept:14,kernel:[0,9,11,13,20,21],keyword:[4,11,20,21],kingma:15,know:[0,1,6,26],knowledg:[0,1],kspace:14,kumar:15,kwarg:[6,11,14],l0constraint:7,l1constraint:7,l_2:1,label:[0,1,8],label_sourc:8,lanczo:[11,20,21],larg:[1,7],larger:[0,11,14,20,21],largest:0,last:[0,1],later:[14,26],latter:[0,1],law:17,layer:[1,19],layout:15,lead:[1,14],leaf:6,least:[0,1],leav:14,left:[1,11,24],legend:0,len:[0,25,26],length:[14,15],less:[14,24,25,26],let:[0,24,25],level:[14,26],leverag:[0,1],libc:22,librari:[0,1,21,22],libstdc:22,lies:1,like:[0,1,3,11,14,15,17,20,21],likelihood:[0,1,13,16,20,21,22,25,26],limit:3,line:[1,13],linear:[0,8,11,24,25],linearmap:[8,24],linearpercentilenorm:8,link:1,lipschitz:[20,21],list:[0,1,4,6,7,8,10,11,19,20,21,25],live:0,load:[1,21,22,23,24,26],local:[1,22,25],locat:[0,1,3,6,11,14,24,26],log:[0,1,7,16,25,26],logarithm:1,logic:[20,21,24,26],logl:[0,25,26],longer:[20,21],longest:24,look:[0,24],lookup:5,loss:[0,4,7,13,25,26],low:[11,13,18,25],lower:[0,1,11,20,21,25],lowr:25,lowresobserv:[13,20,21,25],lpha:17,lsst:[0,20,21,24],lstdc:22,lupton:[1,24],lupton_rgb:[8,24],lust:14,lvl:25,m_k:1,machineri:0,macos_sdk_headers_for_macos_10:22,mactch:13,made:[0,14],magnitud:1,mai:[0,1,22,24,26],main:1,mainli:11,major:14,make:[1,7,9,14,20,21,22,24],make_lupton_rgb:24,make_oper:[20,21],makecatalog:25,mani:[1,14],manifold:1,map:[0,1,3,8,13,20,21,24,25,26],mark:0,markers:25,mask:[1,8,13,14,18],masked_arrai:1,match:[0,1,8,13,18,19,20,21,24,25,26],match_patch:18,match_psf:[9,13],mathbb:1,mathemat:[0,1,7],mathsf:1,matplotlib:[0,1,8,22,24,25,26],matric:14,matrix:[1,14,19,20,21],max:[1,3,25,26],max_it:4,max_pixel:12,maxim:[1,15],maximum:[0,1,4,12],mayb:0,mean:[1,3,4,10,14,19,25],measur:[2,20,21,24],mechan:1,melchior:1,memori:[15,20,21],mention:1,met:1,meta:1,metadata:[0,1,13,20,21],method:[0,1,3,4,5,6,7,8,9,10,11,13,15,16,17,19,20,21,24,25],metric:1,mew:26,might:[9,14,20,21],mimic:24,min:[3,24,26],min_a:19,min_valu:3,mingradi:14,minim:[0,1],minimum:[0,3,14,19,22,24,25,26],minor:1,minu:1,mitchel_netravali:11,mixtur:1,mk_shifter:11,model:[1,3,4,6,7,8,10,12,13,14,19,20,21,23],model_:[0,26],model_fram:[0,13,24,26],model_frame_:0,model_hr:25,model_lr:25,model_psf:[0,26],model_rgb:0,modul:[1,15,20,21,24],moffat:17,moment:15,monoton:[0,7,14,19,20,21,25],monotonicityconstraint:[1,7],more:[0,1,8,14,20,21,24,26],morph:[1,6,7,19],morpholog:[0,1,6,7,11,14,19,20,21,25],most:[0,1,20,21,22],mostli:0,move:22,mse:4,much:[1,14,25],multi:[0,1,8,19,20,21,23],multiband:[0,24,25],multicomponentsourc:[0,1,19,20,21],multipl:[0,1,6,8,9,19,20,21],must:[0,1,22],n_compon:[6,19],n_sourc:[6,19],name:[1,5,10,14,15,20,21],narrow:[1,19],nate:14,nativ:[1,25],navig:22,nbsphinx:22,nbyte:15,ndarrai:15,ndim:15,ndimension:[20,21],nearest:[1,11,14],necessari:[1,11,18,25],need:[0,1,6,7,11,13,14,15,20,21,22,24,25,26],neg:[1,7,24],neighbor:[1,11,14],new_sourc:[0,26],newbyteord:25,newer:22,next:25,node:6,nois:[0,1,7,8,24,26],non:[0,1,6,7,19],none:[0,1,3,6,7,8,9,10,11,13,14,17,19,24,25,26],norm:[0,1,8,24,25,26],normal:[0,8,17,20,21,24],normalizationconstraint:[1,7],notat:0,note:[1,11,24,26],notebook:22,notic:0,now:[0,14,20,21,25],np1:25,np2:25,npz:[0,24,26],number:[0,1,4,6,8,9,10,13,15,19,20,21,25],numer:[8,9],numpi:[0,1,6,8,10,11,14,15,19,22,24,25,26],numpydoc:22,nxn:14,object:[0,1,3,5,6,7,9,11,13,15,17,18,20,21,24,26],obs:25,obs_hdu:25,obs_hsc:25,obs_hst:25,obs_idx:[19,25],observ:[2,3,4,8,14,19,20,21,23,24],obtain:0,odd:[9,20,21],off:[14,24,26],offer:1,offset:14,often:[0,1,7,14,24],old:[20,21],onc:1,one:[0,1,6,13,14,15,20,21],ones:[1,26],ones_lik:[25,26],onli:[1,6,7,13,14,15,17,19],onto:[1,5,7,8,14,24],opaqu:14,open:[0,1,21,24,25],oper:[1,2,3,5,7,20,21,24],oppos:[0,20,21],optim:[1,4,6,14,15,16,20,21,22],optimim:1,optimimz:6,option:[1,13,19,20,21,22,24],orang:0,order:[0,1,7,8,9,11,14,24,25],ordinari:1,organ:1,origin:[0,3,14],other:[0,1,7,13,14,15,25],otherwis:[1,24],our:[0,1,25,26],out:[1,8,24],outer:19,outsid:[1,8,14],over:[0,1,8,11,15,17],overal:[1,19,26],overhaul:[20,21],overlap:[0,1,3],overview:21,overwrit:[20,21],own:1,packag:[0,1,7,20,21,22,25],pad:[9,11,13,14,20,21],page:22,pair:[1,3],param:6,paramet:[2,3,4,6,7,8,9,11,12,13,14,17,18,19,20,21,24,25],parameter:[1,6],parametr:[0,1,6],paremet:11,parikh:1,parramet:16,part:[0,14,15,25],partial:[0,1,26],partner:14,pass:[14,20,21,24],patch:[14,20,21],path:22,pattern:0,peak:[1,14,19,20,21,26],peigen:22,penalti:[1,7,14],per:[0,1],percentil:[8,19],percept:[8,24],perfect:1,perfectli:14,perform:[0,1,11,13,14,17,20,21,26],peril:1,perimet:19,pesquet:1,photometri:26,photutil:0,physic:1,pick:[1,26],pickl:[0,1,20,21,24],piec:1,pip:22,pixel:[0,1,7,8,10,11,12,13,14,17,18,19,20,21,24,25,26],pkg:22,place:[0,11,20,21],plane:14,plot:[0,8,25,26],plt:[0,1,24,25,26],pmelchior:22,pnorm:26,point:[0,1,7,15,19,20,21,23],pointsourc:[0,1,19,20,21,26],poorli:26,popul:1,popular:1,portion:[1,14,25],posit:[6,11,14,19,20,21],positivityconstraint:[1,7],possibl:1,postag:1,poster:0,posteriori:1,potenti:[6,7,20,21,26],power:17,practic:[1,26],pre:0,prefer:[20,21],present:[6,24],preserv:[0,8,24],prevent:[1,7,9,13],previou:14,prgb:26,primari:1,print:[0,1,24,25,26],prior:[0,2,15,20,21,22,26],probabl:[1,14,22],problem:[1,14],proce:[1,24],process:1,produc:8,product:11,profound:1,proj:14,proj_dist:14,project:[1,7,11,14,20,21],project_disk_s:14,project_disk_sed_mean:14,project_imag:11,proper:26,properli:[1,20,21],properti:[1,3,6,7,9,10,17,19],propto:1,prototyp:1,provid:[0,1,4,8,20,21,24],prox_con:14,prox_kspace_symmetri:14,prox_monoton:[7,14],prox_sdss_symmetri:14,prox_soft_symmetri:14,prox_strict_monoton:14,prox_uncentered_symmetri:[7,14],proxim:[1,5,7,20,21,22],proximal_disk_s:14,proxmim:7,proxmin:[1,4,7,20,21,22],psf1:9,psf2:9,psf:[0,1,2,9,10,13,18,19,20,21,24,25,26],psf_hr:13,psf_hsc:25,psf_hst:25,psf_lr:13,psf_match_hr:13,psf_match_lr:13,psf_unmatched_sim:26,pure:1,purpos:1,push:5,pybind11:22,pyplot:[0,1,24,25,26],python:[1,15,20,21,22],quadrat:1,qualiti:[8,24],question:1,quick:[1,21,24,26],quickli:[0,14],quickstart:1,quintic_splin:11,quit:26,rac:17,radial:[0,14],rais:6,ran:[0,25,26],rand:1,random:[0,1,19],randomsourc:[0,1,19],rang:[0,3,7,8,24,25],ratio:0,raycast:14,reach:[1,19],read:0,reader:1,real:[9,11,14,15,20,21],realiz:0,reason:[0,1,22,24],recalcul:[20,21],receiv:22,recent:[0,26],recip:1,recommend:[0,1],reconstruct:[0,1],record:14,recreat:[0,8],rectangular:[1,20,21],red:[0,26],redder:[0,14],reddest:24,reddi:15,reduc:[0,1,14,24],refer:[0,1,5,7,14],refin:25,refit:0,regard:1,regardless:14,regener:9,region:[1,7,14,18,19,20,21,24],rel:[0,1,4,7,8,20,21,22],relative_step:[1,15],releas:[20,21],relev:14,reload:[20,21],remain:[14,26],remaind:1,remov:[20,21],renam:[20,21],render:[0,1,8,13,20,21,24,25,26],reopen:0,repeat:[1,7],replac:[20,21],report:26,repositori:22,repres:[0,1,4,9,10,14,17],represent:[0,1,9],represet:9,reproject:[20,21],requir:[1,9,14,20,21,22],res:25,res_rgb:25,resampl:[2,13,20,21,25],reshap:25,residu:[0,8,25,26],residual_hr:25,residual_lr:25,residual_lr_rgb:25,residual_rgb:0,resiz:[20,21],resolut:[0,1,11,13,18,20,21,23],resourc:1,respect:[0,1,18,25],rest:[1,14],restrict:1,restructur:[20,21],result:[1,3,9,11,13,14,17,21,23],revis:26,rfft:9,rgb:[0,8,20,21,24,25,26],rgb_lr:25,right:[1,11,24],rightarrow:1,rigor:14,ring:26,robert:1,robust:[1,26],rotat:[11,13],roughli:[14,25],row:14,rresolut:13,rule:11,run:[0,20,21,22,25],same:[0,6,9,10,11,13,14,20,21,24,25,26],sampl:[0,11,17,24,26],satisfi:[1,7],save:[20,21],sca:[0,1,24],scalar:13,scale:[1,24,25,26],scarlet:[0,1,2,20,24,25,26],scenc:[8,26],scene:[1,4,10,13,20,21,23],scienc:[0,1],scipi:[14,22],script:22,sdss:[1,14],second:[1,15],sed:[0,1,6,8,14,19,20,21,24],see:[0,1,6,7,11,14,15,19,20,21,24,25,26],seek:[0,1,8],seen:0,seismic:25,select:[13,14,20,21,22],self:[1,3,4,6,7,9,10,13,16,17],sens:14,sensit:1,sep:[0,25],separ:[0,1,8,11,20,21,24,26],sequenti:1,serial:0,sersic:[1,20,21],session:1,set:[0,1,5,6,7,11,13,14,15,18,19,20,21,24,25],set_fram:6,set_titl:[0,24],setup:[0,22,25],sever:[0,1,22],shape:[0,1,3,6,9,10,11,13,14,15,17,18,19,20,21,25,26],shape_hr:18,shape_lr:18,shape_or_box:10,sharp:11,shift:[0,6,11,13,14,19,20,21],shite:14,shortest:[8,24],should:[0,1,8,13,14,20,21,22],show:[0,1,24,25,26],show_observ:[0,8,24,26],show_rend:[0,8,24,26],show_residu:[0,8,24,26],show_s:8,show_scen:[0,8,20,21,24,26],show_sourc:[0,8,20,21,24,26],shown:[8,24],side:13,sigma:[0,17,26],signal:[0,1,8],signatur:[1,6,7,15],similar:[0,1,14,20,21],simpl:[0,1,25],simplest:1,simpli:0,simplifi:[15,20,21],simul:[25,26],simultan:1,sin:13,sinc2d:11,sinc:[9,11,13,25,26],sinc_interp:11,sinc_shift:13,singl:[0,1,6,13,14,19,24,25],singular:[20,21],sinh:[0,24,25],situat:[24,26],size:[3,11,13,14,15,19,20,21,24,25],sky:[0,1],sky_coord:[10,19],slice:[3,11,14,20,21],slices_for:3,slightli:0,slow:1,small:[1,14,20,21],smaller:[0,20,21],smallest:1,smax:[20,21],smooth:1,smoother:1,snr:1,soft:14,soften:24,softwar:24,solut:[1,7,14,16],solv:19,some:[0,1,13,14,15,20,21,22],sort:14,sort_by_radiu:14,soup:1,sourc:[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21,23,24],sources_:0,soution:14,space:[1,9,11,13,14,20,21,24,25],spars:14,sparsiti:1,spatial:[0,1,3,6,9,10,13,19],special:1,specif:[1,24],specifi:[0,1,6,13,14,19,20,21],spectral:[0,1,6,10,13],spectroscop:1,speed:[20,21],sphere:1,sphinx:22,spiral:1,spline:[11,20,21],split:0,spread:1,sprime:25,squar:4,src:[0,1,25,26],stabil:1,stack:[0,20,21,24],stamp:1,standard:[1,17],star:[0,1,26],start:[1,3,8,15,24],state:25,statist:[1,15],std:[1,15],stdlc:22,stellar:[1,26],stem:0,step:[0,7,14,15,20,21,24],still:1,stop:3,store:[1,9,20,21,26],str:0,strength:[7,14],stretch:[0,24,25,26],strict:14,stride:15,string:[14,15,18],strip:[9,20,21],strong:1,stronger:14,strongli:[20,21],structur:0,style:22,sub:[1,3,14,20,21],subclass:[1,20,21],subdivid:11,submanifold:1,subplot:[24,25],subregion:14,subsampl:11,subsample_funct:11,subsequ:19,subspac:1,substanti:1,subtract:[1,14,24],subvers:[20,21],subvolum:[1,6],successfulli:1,sudo:22,suffici:1,suggest:26,suitabl:1,sum:[1,7,24],sum_:1,support:[20,21],supress:9,sure:24,surfac:1,surject:1,swap:25,symmetr:[0,1,7,14,17,19,25],symmetri:[9,14,20,21,26],symmetryconstraint:7,sys:[0,25,26],system:[3,22,26],take:[1,14,20,21,24,25],taken:19,target:[14,20,21,22],tbd:10,telescop:[1,25],tell:9,tensor:[11,13],term:[0,1],test:[1,20,21],test_resampl:25,text:[0,25,26],textrm:24,tfrac:1,than:[0,1,6,9,11,14,20,21,24],thei:[0,1,7,8,22],them:[0,1,14,20,21,24,25,26],theoret:1,therefor:1,thi:[0,1,3,6,7,8,9,10,11,13,14,19,20,21,22,24,25,26],thing:[20,21],think:[0,1],those:[0,1,14],though:[20,21],thought:1,three:[24,26],thresh:[0,7,14,19,26],threshold:[3,7],thresholdconstraint:7,through:1,thu:[20,21,24],tighten:0,tightli:1,time:[0,1,6,9,24,25,26],tini:7,titl:25,too:1,top:[1,8,11],total:[0,15,25,26],track:[20,21],tradit:1,transform:[0,1,5,9,13,14,20,21],transit:1,translat:[0,11],transpar:1,transpos:15,trapezoid:11,travers:[6,15],travi:[20,21],tree:[0,6,19],trim:[11,20,21,25],trim_morpholog:19,truncat:[8,20,21],tupl:[3,6,9,10,11,13,14,15,17,18,19],turn:1,tutori:[0,20,21,24,25],two:[0,1,9,11,14,18,24,25,26],type:[0,1,7,15,17,22,26],typic:1,uint8:8,uncentered_oper:[14,20,21],unchang:14,under:[1,16],undersampl:1,uniform:[0,1,19],union:[13,18],uniqu:[1,24,26],unit:[0,1,3],uniti:17,unless:1,unlik:[14,25],until:19,updat:[1,14,17,20,21],update_dtyp:17,use:[0,1,3,5,7,8,9,11,14,19,20,21,22,24,25,26],use_fft:[20,21],use_nearest:[1,7,14],use_relevant_dim:14,used:[1,3,6,8,9,11,14,15,17,20,21,22,24,25],useful:[0,1,11,22,25],usenearest:14,user:[0,14,20,21,25,26],uses:[0,1,11,14,20,21],using:[0,1,14,20,21,22,24,25],usual:[1,26],util:[0,14],valid:[1,7,16],valu:[1,3,7,8,11,12,13,14,15,19,20,21,24,26],vanish:14,vari:14,varianc:[0,1,24],variat:[8,19],vector:[1,14],veri:[0,1,26],version:[14,20,21,22],vertic:[17,19],vhat:[1,15],vicin:1,view:[6,19],visibl:[0,26],visual:[8,20,21,24,25],vmax:25,vmin:25,voxel:1,wai:[0,1,8,14,22,24,26],wall:[0,25,26],want:[0,1,14,22,24,26],warn:[20,21,22],wavelegth:1,wavelength:[8,14,24],wcs:[1,10,13,20,21,25],wcs_hr:[13,18],wcs_hsc:25,wcs_hst:25,wcs_lr:[13,18],wcs_pix2world:25,wcs_world2pix:25,wdeprec:22,weigh:1,weight:[0,1,8,13,14,19,20,21,23,25,26],weights_hsc:25,weights_hst:25,well:[0,1,7,18,19,22,26],were:[1,20,21],what:[0,1,26],when:[6,7,9,14,15,17,20,21,22],where:[1,7,8,11,14,15,24],whether:[1,3,8,14,15,17],which:[0,1,13,14,17,20,21,22,24,25,26],white:24,whose:11,why:1,wide:[1,7,24],wider:1,width:[1,3,6,8,10,13,14,17,19],window:11,wise:1,within:[1,14],without:[0,1],work:[7,14,22],world:[1,10],worth:26,would:[0,20,21,26],wound:1,write:1,wrong:[20,21],wrt:13,x_i:1,x_j:[1,17],x_window:11,xcode:22,xlabel:[0,25,26],xslice:11,y_i:17,y_window:11,yang:0,yield:[1,7,14],yin:0,ylabel:[0,25,26],you:[0,1,22,24,26],your:[1,22,26],yslice:11,yx0:11,zero:[7,11,13,14]},titles:["Quick Start Guide","Core Concepts","API Documentation","scarlet.bbox","scarlet.blend","scarlet.cache","scarlet.component","scarlet.constraint","scarlet.display","scarlet.fft","scarlet.frame","scarlet.interpolation","scarlet.measure","scarlet.observation","scarlet.operator","scarlet.parameter","scarlet.prior","scarlet.psf","scarlet.resampling","scarlet.source","1.0 (2019-12-22)","<em>scarlet</em> Documentation","Installation","Tutorials","Displaying Scenes","Multi-Resolution Modeling","Point Source Tutorial"],titleterms:{"new":[20,21],For:22,Use:0,addit:[20,21],adjust:24,api:[2,20,21],bbox:3,blend:[1,4,25,26],bug:[20,21],build:22,cach:5,chain:1,chang:[20,21],compon:[1,6],concept:1,constraint:[1,7],core:1,creat:[0,25,26],cube:0,data:[0,25],defin:[0,26],develop:22,displai:[0,8,24,25,26],doc:22,document:[2,21],featur:[20,21],fft:9,filter:24,fit:[0,25,26],fix:[20,21],flux:0,frame:[0,1,10,25,26],full:[0,25],gener:[20,21],get:21,guess:25,guid:0,imag:0,initi:[0,25],instal:22,interact:0,interpol:11,load:[0,25],log:21,measur:[0,12],model:[0,24,25,26],monoton:1,multi:25,normal:1,observ:[0,1,13,25,26],onli:22,oper:14,other:[20,21],overview:1,paramet:[1,15],point:26,posit:1,prior:[1,16],psf:17,quick:0,resampl:18,resolut:25,result:[0,26],save:0,scarlet:[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21,22],scene:[0,24],size:1,sourc:[0,1,19,25,26],start:[0,21],step:1,symmetri:1,tutori:[23,26],user:22,view:[0,25],weight:24}})
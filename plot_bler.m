clear all
close all
clc

%% AWGN CHANNEL BLER

if(1)
    ebno = 0:14;
    
    kan_4 = [0.552978515625, 0.471923828125, 0.3780517578125, 0.3004150390625, 0.2152099609375, 0.14190673828125, 0.0838623046875, 0.0449462890625, 0.02005615234375, 0.006998697916666667, 0.0021447470021802324, 0.0004935315860215054, 8.199976209135692e-05, 9.10610049482793e-06, 6.373807557487869e-07];
    mlp_4 = [0.498779296875, 0.404052734375, 0.3199462890625, 0.22607421875, 0.1474609375, 0.07867431640625, 0.0423583984375, 0.017252604166666668, 0.0061686197916666664, 0.0016297457510964911, 0.000379886864626556, 6.672939823250729e-05, 1.1238980404493002e-05, 2.5514236372376892e-06, 6.139246036934961e-07];
    dizet_4 = [0.6334228515625, 0.55908203125, 0.48828125, 0.4130859375, 0.3212890625, 0.2493896484375, 0.1707763671875, 0.11051432291666667, 0.06768798828125, 0.033935546875, 0.0152587890625, 0.005307404891304348, 0.0016699550914115646, 0.0003387832112517337, 5.373995707682148e-05];
    
    kan_6 = [0.68719482421875, 0.593994140625, 0.48681640625, 0.3817138671875, 0.27130126953125, 0.175445556640625, 0.095855712890625, 0.048370361328125, 0.019327799479166668, 0.006351470947265625, 0.0016632080078125, 0.0003175932785560345, 4.658272595909553e-05, 4.532725943909553e-06, 2.935932709472651e-07];
    mlp_6 = [0.650970458984375, 0.55712890625, 0.451995849609375, 0.338897705078125, 0.23138427734375, 0.139373779296875, 0.0767822265625, 0.036102294921875, 0.013458251953125, 0.004494406960227273, 0.0012190969366776315, 0.0002810238329179448, 5.801820936311787e-05, 5.974573510963909e-6, 3.144076961423637e-7];
    dizet_6 = [0.731048583984375, 0.658233642578125, 0.577301025390625, 0.48089599609375, 0.383331298828125, 0.284912109375, 0.195526123046875, 0.11578369140625, 0.06500244140625, 0.029947916666666668, 0.011484781901041666, 0.003629796645220588, 0.0008741106305803571, 0.00016504236169763512, 1.989109896681596e-05];
    
    fig1 = figure(1)
    bk_4 = semilogy(ebno, kan_4, "r.-")
    hold on
    grid on
    box on
    bm_4 = semilogy(ebno, mlp_4, "m.--")
    bd_4 = semilogy(ebno, dizet_4, "b.:")
    
    bk_6 = semilogy(ebno, kan_6, "rsquare-")
    bm_6 = semilogy(ebno, mlp_6, "msquare--")
    bd_6 = semilogy(ebno, dizet_6, "bsquare:")
    
    legend([bk_4, bm_4, bd_4, bk_6, bm_6, bd_6], {"KAN (K=4)", "MLP (K=4)", "DiZeT (K=4)", "KAN (K=6)", "MLP (K=6)", "DiZeT (K=6)"}, location = "south west")
    xlabel("Eb/N0")
    ylabel("BLER")
    hold off
    saveas(fig1, "awgn_bler.eps", eps)
end

%% FADE CHANNEL BLER

if(1)
    ebno = 0:2:40;
    
    kan_4 = [0.609130859375, 0.5009765625, 0.388427734375, 0.2901611328125, 0.2025146484375, 0.137451171875, 0.09098307291666667, 0.05926513671875, 0.039013671875, 0.0252532958984375, 0.016031901041666668, 0.010179307725694444, 0.00635186557112069, 0.004194779829545455, 0.002638462611607143, 0.0015691773504273505, 0.0009984555451766305, 0.0006313981681034483, 0.0004183277147545662, 0.0002638407330691643, 0.00015949953723867595];
    mlp_4 = [0.5670166015625, 0.4571533203125, 0.34716796875, 0.255126953125, 0.173583984375, 0.11944580078125, 0.0792236328125, 0.04931640625, 0.031962076822916664, 0.020792643229166668, 0.012394205729166666, 0.00839510830965909, 0.004994985219594595, 0.003411187065972222, 0.0020399305555555557, 0.0013472285583941606, 0.0008366554295091324, 0.0005216680021367521, 0.0003274100167410714, 0.00020046776702680526, 0.00012515753161312373];
    dizet_4 = [0.6610107421875, 0.575439453125, 0.467529296875, 0.36376953125, 0.25958251953125, 0.17974853515625, 0.12263997395833333, 0.08184814453125, 0.05352783203125, 0.03310546875, 0.02119140625, 0.01413241299715909, 0.00888671875, 0.005723741319444444, 0.0036175896139705884, 0.0023463591364503815, 0.001454380580357143, 0.0009245627596299094, 0.000569585543959888, 0.00035791072867819463, 0.00022390005961115187];
    
    kan_6 = [0.70819091796875, 0.588775634765625, 0.465057373046875, 0.342987060546875, 0.24102783203125, 0.16839599609375, 0.106475830078125, 0.073333740234375, 0.0462493896484375, 0.0294647216796875, 0.018768310546875, 0.01190948486328125, 0.007904052734375, 0.00479736328125, 0.0030269622802734375, 0.0019480387369791667, 0.0012231124074835527, 0.00077362060546875, 0.0004918524559507978, 0.00030968173238255035, 0.00018838011188271605];
    mlp_6 = [0.6845703125, 0.56591796875, 0.4378662109375, 0.32281494140625, 0.226593017578125, 0.154022216796875, 0.10504150390625, 0.0660400390625, 0.0416717529296875, 0.0278472900390625, 0.017527262369791668, 0.01146697998046875, 0.006801060267857143, 0.004425048828125, 0.002814797794117647, 0.0018169696514423077, 0.0011298947217987805, 0.0007393129410282258, 0.0004556485922029703, 0.00028135718368902437, 0.00016936692804428045];
    dizet_6 = [0.751556396484375, 0.640869140625, 0.5225830078125, 0.396728515625, 0.293975830078125, 0.20574951171875, 0.1385498046875, 0.090301513671875, 0.058441162109375, 0.03839111328125, 0.0246429443359375, 0.015460205078125, 0.00997161865234375, 0.006312443659855769, 0.00386962890625, 0.0024347305297851562, 0.0015926361083984375, 0.0009647369384765625, 0.0006184731760332661, 0.00040170769942434213, 0.0002631878030711207];
    
    fig2 = figure(2)
    bk_4 = semilogy(ebno, kan_4, "r.-")
    hold on
    grid on
    box on
    bm_4 = semilogy(ebno, mlp_4, "m.--")
    bd_4 = semilogy(ebno, dizet_4, "b.:")
    
    bk_6 = semilogy(ebno, kan_6, "rsquare-")
    bm_6 = semilogy(ebno, mlp_6, "msquare--")
    bd_6 = semilogy(ebno, dizet_6, "bsquare:")
    
    legend([bk_4, bm_4, bd_4, bk_6, bm_6, bd_6], {"KAN (K=4)", "MLP (K=4)", "DiZeT (K=4)", "KAN (K=6)", "MLP (K=6)", "DiZeT (K=6)"}, location = "south west")
    xlabel("Eb/N0")
    ylabel("BLER")
    hold off
    saveas(fig2, "fade_bler.eps", eps)
end
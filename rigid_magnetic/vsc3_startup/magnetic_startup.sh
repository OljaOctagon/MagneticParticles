for irun in 1 2 3 4 5 6 7 8
do
	for shift in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8
	do
		for li in 1 1.5 2 2.5 3 3.5 4 4.5 5 6 7 8 9 10 15 20 25 30 35 40 50 100 
		do
			dir="mag2p_shift_"$shift"_lambda_"$li"_phi2d_0.0106_rid_"$irun
		        mkdir $dir
			cp in.mag2patch-quasi-2d $dir	
			cp 2patch.txt $dir 

			mu_squared=0.01
			Vp=0.5235987755982988
	  		temp=$(echo "scale=9; $mu_squared/($li*$Vp)" | bc)	
			cp runlammps.sh $dir
			sed -i "s/Temperature/$temp/" $dir/runlammps.sh
		done
	done
done

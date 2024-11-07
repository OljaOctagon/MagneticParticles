for irun in 1 
do
	for shift in 0 0.2 0.4 0.8
	do
		for li in 1 6 100 
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

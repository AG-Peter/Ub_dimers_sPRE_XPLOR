proc make_trajectory_movie_files {} {
	# set num [molinfo top get numframes]
    set num 200
	# loop through the frames
	for {set i 0} {$i < $num} {incr i} {
		# go to the given frame
		animate goto $i
                # force display update
                display update
		# take the picture
        puts "Saving frame $i"
		# take_picture
        mol on 1
        render STL /mnt/data/kevin/xplor_analysis_files/cluster_analysis_cog_dihe/k6/cluster_2/k6_cluster_2_$i.stl
        mol off 1
        mol on 2
        render STL /mnt/data/kevin/xplor_analysis_files/cluster_analysis_cog_dihe/k29/cluster_0/k29_cluster_0_$i.stl
        mol off 2
        mol on 3
        render STL /mnt/data/kevin/xplor_analysis_files/cluster_analysis_cog_dihe/k29/cluster_3/k29_cluster_3_$i.stl
        mol off 3
        mol on 4
        render STL /mnt/data/kevin/xplor_analysis_files/cluster_analysis_cog_dihe/k29/cluster_4/k29_cluster_4_$i.stl
        mol off 4
        mol on 5
        render STL /mnt/data/kevin/xplor_analysis_files/cluster_analysis_cog_dihe/k33/cluster_3/k33_cluster_3_$i.stl
        mol off 5
        mol on 6
        render STL /mnt/data/kevin/xplor_analysis_files/cluster_analysis_cog_dihe/k33/cluster_4/k33_cluster_4_$i.stl
        mol off 6
    }
}

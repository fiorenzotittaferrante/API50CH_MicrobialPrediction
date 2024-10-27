use File::Basename;

$dirname = '../../Data/BacDive/ncbi/Decompress/';
$dirname_AFLP = '../../Data/BacDive/AFLP_perl';

opendir(DIR, $dirname) or die "Could not open $dirname\n";

while ($file = readdir(DIR)) {
	next if ($file eq ".");
	next if ($file eq "..");
	
	$filename = basename($file, ".fna");
	print "$filename\n";
	system("perl AFLPinSilico_v16.pl -i $dirname$filename.fna -rs1 'G^AATTC' -rs2 'T^TAA' -p AFLP");
	system("ren AFLP_inSilico_'GAATTC'_'TTAA'.wri $filename.wri");
	system("move $filename.wri $dirname_AFLP");
}

closedir(DIR);

$dirname_Patric = '../../Data/BacDive/Patric/';
opendir(DIR, $dirname_Patric) or die "Could not open $dirname\n";

while ($file = readdir(DIR)) {
	next if ($file eq ".");
	next if ($file eq "..");
	
	$filename = basename($file, ".fna");
	print "$filename\n";
	system("perl AFLPinSilico_v16.pl -i $dirname_Patric$filename.fna -rs1 'G^AATTC' -rs2 'T^TAA' -p AFLP");
	system("ren AFLP_inSilico_'GAATTC'_'TTAA'.wri $filename.wri");
	system("move $filename.wri $dirname_AFLP");
}

closedir(DIR);
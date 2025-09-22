#!/usr/local/bin/perl -w

=head1 Description

version 1.4 (05/04/2004)

change the way to input paramters (minor)
usage: 
 AFLPinSilico.pl -i <multi-fasta file> -rs1 <restriction site> -rs2 <restriction site> -adpt <adapt length> -sn1 <5'selective base> -sn2 <3'selective base> -p <options>

version 1.3 (05/09/2003)

usage: AFLPinSilico.pl <multi-fasta file> <restriction site> <restriction site> <adapt length> <5'selective base> <3'selective base> <options>


-----------------------------------------------------------------------------------------------------------------------------------------

e.g. perl AFLPinSilico.pl -i example/test_seq.tfa -rs1 'TTAA' -rs2 'CATG' -adpt 20 -sn1 N -sn2 N -p TPF

	  perl AFLPinSilico.pl -i example/test_seq.tfa -rs1 'CCGG' -rs2 'TTAA' -sn1 list:example/selective_nucl.list -p AFLP

output file = TPF_inSilico_CCGGG_TTTAA.wri
	 
	 remarks:
	 	UIPAC code can be used for restriction sites as well as selective base
		Takes as input a directory of files in fasta or a multi-fasta file
		param5 and param6 can be replaced by a TAB-delimited list, syntax: 'list:filename' (1 combination per line)
		order by which restriction enzymes are important!
				
	 options: (either -TPF or -AFLP has to be set)
	 	-TPF for Transcript profiling
		-AFLP for Amplified Fragment Length Polymorfisme
		-HELP
		-VERB for more on screen text...

Reference to cite when using this applcation:

"AFLPinSilico, simulating AFLP fingerprints" (2003)
Stephane Rombauts, Yves Van de Peer, and Pierre RouzÈ
Bioinformatics. 2003 Apr 12;19(6):776-777,PMID: 12691992.


copyrigth: (2002-2003)

 Vlaams Instituut voor Biotechnologie (VIB)
 DEPARTMENT OF PLANT SYSTEMS BIOLOGY
 Research group of Bioinformatics

 Stephane Rombauts				tel.:32(0)9 3313821
 GHENT UNIVERSITY, 				Fax :32(0)9 3313800
 Technologie Park 927, 
 B-9052 Gent, 
 Belgium
 mailto:strom@psb.ugent.be  		http://www.psb.ugent.be

=cut

#	foreach i ( /group/biocomp/Genome/Arabidopsis/Ceres/AllEntries/*.tfa )
#	echo $i
#	AFLPinSilico.pl $i RGATCY TTAA 22 A T >> BstYI_polyA.list
#	end
#
#========================================================================================
use lib "./";
use AFLPinSilico_strom;

use CGI;
use strict;
use Getopt::Long;

#========================================================================================
#-----------------------------------------------------------------------------------------
my ($input,$site1,$site2,$adaptorLength,$selectiveNT1,$selectiveNT2,$program,$help,$verbose)=('','','',0,'','','',0,0);

GetOptions(
	"i=s"  => \$input,
	"rs1=s" => \$site1,
	"rs2:s" => \$site2,
	"adpt:i" => \$adaptorLength,
	"sn1:s" => \$selectiveNT1,
	"sn2:s" => \$selectiveNT2,
	"p=s" => \$program,
	"help+" => \$help,
	"verb+" => \$verbose
) or &usage("not enough parameters");
chomp($input);
chomp($site1);
chomp($site2);
chomp($adaptorLength);
chomp($selectiveNT1);
chomp($selectiveNT2);
chomp($program);
chomp($help);

my	$fastaFile = "";
my	$fasta_dir = "";
my	@fasta_files = ();
my	@select_nucl = ();
my	$real_site1 = '';
my	$real_site2 = '';

if ($help)
{
	&usage("help:\n\n");
}
else
{	
	if (!-d $input)
	{
		$fastaFile = $input;
		@fasta_files = $input;
	}
	else
	{
		$fasta_dir = $input;
		$fasta_dir =~ s/\/$//;

		# retrieve fasta files
		@fasta_files = glob("${fasta_dir}/*.tfa") || die "can't retrieve *.tfa files\n";
		
		if (scalar(@fasta_files) < 1)
		{
		    die "no FASTA files (*.tfa) in $fasta_dir";
		}
	}

	if(defined($site1) || length($site1) >= length($site2))
	{
		$real_site1 = uc($site1);
		$real_site1 =~ s/\^//;
		$real_site2 = uc($site2);
		$real_site2 =~ s/\^//;
		
		if($selectiveNT1 =~ m/list:(\S+)/i)
		{
			my $x=0;
			my $s1='';
			my $s2='';
			open(LIST, "< $1") || die "can't find list:$1\n";
			while(<LIST>)
			{
				($s1,$s2) = split("[\t ]");
				chomp($s1);
				chomp($s2);
				$select_nucl[$x]->{'selectiv5'} = uc($s1);
				$select_nucl[$x]->{'selectiv3'} = uc($s2);
				$x++;
			}
			close(LIST);
		}
		else
		{
			$select_nucl[0]->{'selectiv5'} = uc($selectiveNT1);
			$select_nucl[0]->{'selectiv3'} = uc($selectiveNT2);
		}
	}
	else
	{
		$real_site1 = uc($site2);
		$real_site1 =~ s/\^//;
		$real_site2 = uc($site1);
		$real_site2 =~ s/\^//;
		
		if($selectiveNT1 =~ m/list:(\S+)/i)
		{
			my $x=0;
			my $s1='';
			my $s2='';
			open(LIST, "< $1") || die "can't find list:$1\n";
			while(<LIST>)
			{
				($s1,$s2) = split("[\t ]");
				chomp($s1);
				chomp($s2);
				$select_nucl[$x]->{'selectiv5'} = uc($s2);
				$select_nucl[$x]->{'selectiv3'} = uc($s1);
				$x++;
			}
			close(LIST);
		}
		else
		{
			$select_nucl[0]->{'selectiv5'} = uc($selectiveNT2);
			$select_nucl[0]->{'selectiv3'} = uc($selectiveNT1);
		}
	}
}
#----------------------------------------------------------------------------
#-----MAIN-------------------------------------------------------------------

my %restriction_hash = &motif2hash($real_site1,$real_site2,\@select_nucl);

	my $head = "Stephane Rombauts, Yves Van de Peer and Pierre Rouze\n";
	 	$head .= "AFLPinSilico, simulating AFLP fingerprints \n";
	 	$head .= "Bioinformatics. 2003 Apr 12;19(6):776-7. PMID: 12691992 \n\n";
		$head .= "time of execution:" .  localtime() . "\n\n";
		$head .= "file or dir:\t\t" . $input . "\n";
		$head .= "restriction site:\t\t" . $site1 . "\n";
		$head .= "restriction site:\t\t" . $site2 . "\n";
		$head .= "adaptorLength:\t\t" . $adaptorLength . "\n";
		$head .= "selective nulc.:\t\t" . $selectiveNT1 . "\n";
		$head .= "selective nulc.:\t\t" . $selectiveNT2 . "\n\n\n";
#		$head .= "$real_site1 = ${restrictionSite1}-${restrictionSite2} = $real_site2\n\n\n";

	my $i = 0;
	my $num_fragments=0;
	my $num_tpf	=0;
	my %sequence_file = ();
	my $counter = 1;
	 
	foreach my $f (@fasta_files)
	{
		$f =~ s/\\/\//g;
		print "\n-----------------------------------$f\n" if($verbose);
		 
		%sequence_file = &fasta2hash($f);
		 
		##-------------------------------------------------
		if ($program =~ m/TPF/i)
		{
			open (TPF, "> TPF_inSilico_${real_site1}_${real_site2}.wri");
			print TPF $head;
	  		
			my @TPF = &transcript_profiling(\%sequence_file, $adaptorLength, $real_site1, $real_site2, \%restriction_hash);

			my @TPFtemp = ();
			$counter = 1;
			@TPFtemp = sort {$a->{'fragLength'} <=> $b->{'fragLength'}} @TPF;
			print TPF "\tfile\tseqLength\t${site1}\t${site2}\tFragLength\tFragment\n";
			foreach my $thing (@TPFtemp)
			{
				print TPF "$counter\t$thing->{'name'}\t$thing->{'seqLength'}\t$thing->{'pos1'}\t$thing->{'pos2'}\t$thing->{'fragLength'}\t$thing->{'fragment'}\n" ;
				$counter++;
			}
			@TPFtemp = ();
			@TPF = ();
			close (TPF);
		}
	##-------------------------------------------------
		if ($program =~ m/AFLP/i)			
		{
			open (AFLP, "> AFLP_inSilico_${real_site1}_${real_site2}.wri") ;
			print AFLP $head;
	  		
			my @AFLP = &AmplifiedFragmentLenghtPolymorphism(\%sequence_file, $real_site1, $real_site2, \%restriction_hash);

			my @AFLPtemp = ();
			$counter = 1;
			if(scalar(@AFLP)>0) { @AFLPtemp = sort {$a->{'fragLength'} <=> $b->{'fragLength'}} @AFLP; }
			print AFLP "\tfile\tseqLength\t5' site\t\t3' site\t\tFragLength\tFragment\n";
			foreach my $thing (@AFLPtemp)
			{
				print AFLP "$counter\t$thing->{'name'}\t$thing->{'seqLength'}\t$thing->{'site1'}\t$thing->{'pos1'}\t$thing->{'site2'}\t$thing->{'pos2'}\t$thing->{'fragLength'}\t$thing->{'fragment'}\n";
				$counter++;
			}
			@AFLPtemp = ();
			@AFLP = ();
			close (AFLP) 
		}
	##-------------------------------------------------
		if ($program =~ m/ROEL/i)			
		{
			open (AFLP, "> AFLP_inSilico_${real_site1}_${real_site2}.wri") ;
			print AFLP $head;
	  		
			my @AFLP = &AmplifiedROELFragmentPolymorphism(\%sequence_file, $real_site1, $real_site2, \%restriction_hash);

			my @AFLPtemp = ();
			$counter = 1;
			if(scalar(@AFLP)>0) { @AFLPtemp = sort {$a->{'fragLength'} <=> $b->{'fragLength'}} @AFLP; }
			print AFLP "\tfile\tseqLength\t5' site\t\t3' site\t\tFragLength\tFragment\n";
			foreach my $thing (@AFLPtemp)
			{
				print AFLP "$counter\t$thing->{'name'}\t$thing->{'seqLength'}\t$thing->{'site1'}\t$thing->{'pos1'}\t$thing->{'site2'}\t$thing->{'pos2'}\t$thing->{'fragLength'}\t$thing->{'fragment'}\n";
				$counter++;
			}
			@AFLPtemp = ();
			@AFLP = ();
			close (AFLP) 
		}
	##-------------------------------------------------
		if ($program =~ m/SITE/i)
		{
			my @site_stat = &sitestat(\%sequence_file, $real_site1);	

			open (SITE, "> SITE_inSilico_${real_site1}.wri") ;
			$head .=  "seq\tnumber\tPos.site\tfragLength\tseqLength\tfreq\n";
			print SITE $head;
			foreach my $x (@site_stat)
			{
				print SITE $x;
			} 
			close (SITE);
		}
	##-------------------------------------------------
		if ($program =~ m/DIGEST/i)
		{
			my @digest = &Digest(\%sequence_file, $real_site1, $site1);	

			open (SITE, "> DIGEST_inSilico_${real_site1}.wri") ;
			$head .=  "source\tenzyme\tsite\tpos\tfragLength\tfrag\n";
			print SITE $head;
			foreach my $frag (@digest)
			{
				#my $frag = $$frag{'frag'};
				#$frag =~ s/$real_site1/$site1/g;
				print SITE join("\t", ($$frag{'source'},$$frag{'enzyme'},$$frag{'pos'},length($$frag{'frag'}),$$frag{'frag'})) ."\n";
			} 
			close (SITE);
		}
	##-------------------------------------------------
	}#foreach my $f (@fasta_files)

#}#foreach my $selectivNucletide (@select_nucl)
exit;



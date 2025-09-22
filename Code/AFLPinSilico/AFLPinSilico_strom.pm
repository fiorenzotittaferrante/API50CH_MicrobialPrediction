package AFLPinSilico_strom;

use strict;
use Digest::MD5;

=head1 Description

  Contains some usefull functions such as

=head2 reverse_seq(seq)

  seq is a string, this function returns another string with all characters in reverse order

=head2 complement(seq)

  seq is a flat nucleic acid sequence stored in a string, 
  this function returns the complement of the sequence (in uppercase)

=head2 translate(seq)

  seq is a flat nucleic acid sequence stored in a string, 
  this function returns the proteic translation of the sequence upto the first stop (in uppercase)

=head2 full_translate(seq)

  seq is a flat nucleic acid sequence stored in a string, 
  this function returns the proteic translation of the full sequence (in uppercase)

=head2 fasta2flat(file) fasta2flat(file, AC)

  file is the path of a FASTA formated file.
  This function returns the first flat sequence of this file or the one with the given 
  accession number AC (an accession number is a valid identifier WITHOUT internal blanks)

=head2 flat2fasta(file, AC, seq, line_length)

  This function APPENDS a sequence to a FASTA formatted file (file), 
  creates a header line with AC and formats the sequence (seq) with line_length375
  characters per lines

=head2 fasta_cut(file, begin, end)

  This function reads a ONE ENTRY FASTA file and return the sub sequence from begin to end
  
=head2 fasta_length(file)

  This function reads a ONE ENTRY FASTA file and return the length of the sequence

=head2 flat2gb(file, LOCUS, seq)

  This function CREATE a GenBank formated file containing the sequence
  given as parameter, and labelled with LOCUS.

=head2 fasta2hash(file)

  Reads in the given (multiple) fasta file and returns a hash in which the keys are the 
  identifiers of the identifiers of the sequences (without the '>') and the values are 
  the sequences themselves. Created by cesim 14/01/2002

=head2 reverseComplement ( sequence )

	Reads a sequence as a continuous string of DNA and returns the reverse-complement in the same format

=head2 hash2fasta( file_name, fasta_hash )

	reads a fasta_hash given by fasta2hash and writes the hash into a fasta formated file named 'file_name'

=head2 readfastas( name )	

	reads directory of fasta files or a fasta file. Returns an array of the file(s) names.

=head2 blast2hash ( blast_file_name )

	Reads a blast output file in default format and returns a hash of the blast. Also concatenated multiple blast output.
	example of usage:
	"my $fileIN =$ARGV[0]; 

	my %BLAST = &blast2hash($fileIN);
	my $hash_of_hits;
	my $hash_of_details;

	foreach my $blast_query (keys(%BLAST))
	{
		print "$blast_query\n";
		$hash_of_hits = $BLAST{$blast_query}{'hits'};
		foreach my $blast_hit (keys(%$hash_of_hits))
		{
			print "\t$blast_hit\n";
			$hash_of_details = $$hash_of_hits{$blast_hit};
			my @sort_keys = (sort { $$hash_of_hits{$b}[0]{'bitscore'} <=> $$hash_of_hits{$a}[0]{'bitscore'} } keys(%$hash_of_hits) );
			foreach my $blast_hit_detail (@sort_keys)
			{
				print "\t$blast_hit_detail->{'Frame'}\t";
				print "\t$blast_hit_detail->{'Evalue'}\n\n";
			}
		}	
	}"

=head2 blast2hash2 ( blast_file_name )

	Reads a blast output file in default format and returns a hash of the blast. Also concatenated multiple blast output.
	example of usage:
	"my $fileIN =$ARGV[0]; 

	my %BLAST = &blast2hash($fileIN);
	my $hash_of_hits;
	my $hash_of_details;

	foreach my $blast_query (keys(%BLAST))
	{
		print "$blast_query\n";
		$ARRAY_of_hits = $BLAST{$blast_query}{'hits'};
		foreach my $blast_hit (@$hash_of_hits)
		{
			print "\t$blast_hit{'hitACnr'}\n";
			$hash_of_details = $$hash_of_hits{$blast_hit};
			my @sort_keys = (sort { $$hash_of_hits{$b}[0]{'bitscore'} <=> $$hash_of_hits{$a}[0]{'bitscore'} } keys(%$hash_of_hits) );
			foreach my $blast_hit_detail (@sort_keys)
			{
				print "\t$blast_hit_detail->{'Frame'}\t";
				print "\t$blast_hit_detail->{'Evalue'}\n\n";
			}
		}	
	}"

=head2 longestORF ( DNA_sequence )
	
	Reads a flat sequence and looks for the longest ORF (from start-stop). Returns a tab-delimited string with UTR5-CDS-UTR3.

=head2 readEuGene ( prediction_file DNA_sequence )
	
	Reads the EuGene output and a strin containing a flat sequence. Returns a array of hash with keys (strand,exons,phase,frame,  .

=head2 generateFROMmotif( $ \@ )

	Iteratif procedure that reads a motif as input a motif (DNA), and returns all the possible sequences
	represented by this motif as a list. Degenerate code is allowed and that numbers always have to be set
	between brackets, e.g. AATGNW(2,6)TTT, AATGNW(2)TTTRVH or AATGTTT
	warning: be aware that a very complex motif can generate an extensive list of
	possibilities, which will block any computer.

=cut
  
BEGIN {
  use Exporter ();
  use vars qw($VERSION @ISA @EXPORT);
  
  # set version
  $VERSION = 1.00;
  
  @ISA = qw(Exporter);
  @EXPORT = qw(&reverse_seq 
  			&complement 
			&reverseComplement 
			&makeIUPAC 
			&fasta2hash 
			&fastahandle2hash
			&usage
			&error_handeling
			&generate
			&motif2hash
			&transcript_profiling
			&AmplifiedFragmentLenghtPolymorphism
			&AmplifiedROELFragmentPolymorphism
			&sitestat
			&Digest
			&makeNR
			);
}

#use vars @EXPORT_OK;


#------------------------------------------------------------
# reverse a flat sequence
sub reverse_seq ( $ )
  {
    my ($seq) = @_;

    return join('', reverse (split '', $seq));
  }

#------------------------------------------------------------
# complement a flat sequence
sub complement( $ )
  {
    my ($seq) = @_;
    
    $seq = uc($seq);
    $seq =~ tr/('A','T','G','C')/('T','A','C','G')/;
    return $seq;
  }
#-----------------------------------------------------------------------------------------
# created by Stephane Rombauts (strom 12/08/2003)

sub reverseComplement ( $ ) 
{
   my $tmp_sequence = $_[0];
	my %complement = ("A" => "T",
	       "T" => "A",
	       "C" => "G",
	       "G" => "C",
			 "a" => "t",
			 "t" => "a",
			 "c" => "g",
			 "g" => "c",
			 "M" => "K",
			 "R" => "Y",
			 "W" => "W",
			 "S" => "S",
			 "Y" => "R",
			 "K" => "M",
			 "V" => "B",
			 "H" => "D",
			 "D" => "H",
			 "B" => "V",
			 "m" => "k",
			 "r" => "y",
			 "w" => "w",
			 "s" => "s",
			 "y" => "r",
			 "k" => "m",
			 "v" => "b",
			 "h" => "d",
			 "d" => "h",
			 "b" => "v",
			 "N" => "N",
			 "X" => "X",
			 "n" => "n",
			 "x" => "x",
			 "^" => "^");
	my $CDS_comp = "";
	my $Len = 0;
	$Len = length($tmp_sequence) if(defined($tmp_sequence));
	$tmp_sequence = reverse($tmp_sequence) if(defined($tmp_sequence));
	
	for (my $j=0; $j < $Len ; $j++)
	{
		if(!exists $complement{substr($tmp_sequence,$j,1)}) { printf STDERR " no complement for: " . substr($tmp_sequence,$j,1) . "\n"; }
		$CDS_comp .= $complement{substr($tmp_sequence,$j,1)};
	}
	return($CDS_comp);
}

#------------------------------------------------------------
# transform UIPAC code into simple 4 letter code
sub makeIUPAC( $ )
{
	my ($original) = @_;
	my $i = 0;
	my %IUB = ("A" => "A",
	       "C" => "C",
	       "G" => "G",
	       "T" => "T",
	       "U" => "T",
	       "M" => "(A/C)",
	       "R" => "(A/G)",
	       "W" => "(A/T)",
			 "S" => "(C/G)",
			 "Y" => "(C/T)",
			 "K" => "(G/T)",
			 "V" => "(A/C/G)",
			 "H" => "(A/C/T)",
			 "D" => "(A/G/T)",
			 "B" => "(C/G/T)",
			 "N" => "(A/C/G/T)",
			 "X" => "(A/C/G/T)",
			 "(" => "(",
			 ")" => ")",
			 "," => ",",
			 "/" => "/",
			 "1" => "1",
			 "2" => "2",
			 "3" => "3",
			 "4" => "4",
			 "5" => "5",
			 "6" => "6",
			 "7" => "7",
			 "8" => "8",
			 "9" => "9",
			 "0" => "0"
				);
	my $IUPACsite ='';
	
	for ($i=0; $i < length($original) ; $i++)
	{
		$IUPACsite .= $IUB{uc(substr($original,$i,1))};
	}

	return($IUPACsite);
}
#------------------------------------------------------------
# Reads in an entire (multiple) fasta file and returns a hash in which
# the keys are the identifiers of the sequences (without the '>')
# and the values are the sequences themselves. Created by cesim 14/01/2002
#	modified by Stephane Rombauts (strom 12/08/2003)

sub fasta2hash ( $ )
 {
  my ($file,$key,$value);
  my (%fasta_hash);
  $file=$_[0];
  open (IN,$file) || die &usage("problem with $file\n");
  while (<IN>)
   {
    chomp;
    if (/^>(\S+)\s*(.*)/)
     {
	 my @rec = split(m/\|/,$1);
	 while(scalar(@rec)>0)
	 {
      	$key = pop(@rec);
		last if(length($key)>3);
	 }
      	#$fasta_hash{$key}{"comment"}=$2;
		#if(defined($fasta_hash{$key}{"sequence"}) || $fasta_hash{$key}{"sequence"} ne '')
		#{
		#	print STDERR "sequence $key already exists\n";
			$fasta_hash{$key}='';
		#}
     } #if (/^>(\w)$/)
    else 
     {
      $key || die "File $file is not a fasta file!\n$key\n$_\n";
      s/\s+//g;
      $fasta_hash{$key}.=uc($_);
     } #else 
   } #while (<IN>)
  close IN;
  return (%fasta_hash); 
 } #fasta2hash ( $ )

#-----------------------------------------------------------------------------------------
sub fastahandle2hash ( $ )
 {
  my ($file,$key,$value);
  my (%fasta_hash);
	my $chr13 = chr(13);
	my $chr0 = chr(0);
  $file=$_[0];
#  open(IN, $file);
  while (<$file>)
   {
    	chomp;
		$_ =~ s/$chr13//g;
		$_ =~ s/$chr0//g;
    	if (/^>(\S+)/)
     	{
      	$key=$1;
     	} #if (/^>(\w)$/)
    	else 
     	{
      	$key || die "File $file is not a fasta file!";
      	s/\s+//g;
      	$fasta_hash{$key}.=uc($_);
     	} #else 
   } #while (<IN>)
  return (%fasta_hash); 
 } #fasta2hash ( $ )

#========================================================================================
#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
sub error_handeling ( $ $ )
{
	print STDERR "  $_[0] \n\n $_[1] \n";
	exit;
}
#-----------------------------------------------------------------------------------------
sub usage ( $ )
  {
    print STDERR "$_[0]\n";
    system("pod2text $0");
    exit(1);
  }
#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
# created by Stephane Rombauts (strom 12/08/2003)
#	it takes as input a motif (DNA), and returns all the possible sequences
#	represented in this motif as a list.
#
#	e.g. GenerateMotifs.pl 'AATGNW(2)TTT'
#	e.g. GenerateMotifs.pl 'AATGNW(2)TTTRVH'
#	e.g. GenerateMotifs.pl 'AATGNW(2,6)TTT'
#	e.g. GenerateMotifs.pl 'AATGTTT'
#
#	note that degenerate code is allowed and that numbers always have to be set
#	between brackets and to avoid any problems also put the motif between quotes.
#
#	warning: be aware that a very complex motif can generate an extensive list of
#	possibilities, which will block any computer.
#
#	created 23/03/2001
#	modified 23/03/2001

sub generate( $ \@ )
{
	my ($pattern, $result) = @_;
	if ($pattern eq "" ) {return; }
	@{$result}=();

	$pattern =~ s/N/\(A\/C\/G\/T\)/g;	
#   print "\npattern $pattern, result $result \n";
#   print ".";

	my ($fix, $choice, $min, $rest);
	my $max = 0;

	if ($pattern =~ m/[0-9]/)
	{
		#split at the same time things like this  (9) and (6,9) like it is now (6,9)thing works
		($fix, $choice, $min, $max, $rest) = split(/(\([ACGT\/]+\))\((\d+),(\d+)\)(.*)$/i, $pattern);
# print "\nfix $fix choice $choice, time $min, max $max rest $rest";
		if (!defined($max) || $max !~ m/[0-9]/)
		{
			($fix, $choice, $max, $rest) = split(/(\([ACGT\/]+\))\((\d+)\)(.*)$/i, $pattern);
			$min = 0;
# print "\nfix $fix choice $choice, time $min, max $max rest $rest";
		}
		if (defined($fix) && $fix =~ m/[0-9]/)
		{
			($fix, $min, $rest) = split(/\((\d+)\)(.*)$/i, $pattern);
			if (defined($fix) && $fix =~ m/[0-9]/)
			{
				($fix, $min, $max, $rest) = split(/\((\d+),(\d+)\)(.*)$/i, $pattern);
			}
			else
			{
				$max = $min;
			}
			$choice = chop($fix);
#print "\n again: fix $fix choice $choice, time $min, max $max rest $rest";
		}
		if (defined($fix) && defined($choice) && defined($min) && defined($max) && defined($rest))
		{
#print "\n and again:fix $fix choice $choice, time $min, max $max rest $rest";
			my (@temp, $x, $i);
			for ($x=$min; $x<=$max ; $x++)
			{
#				print "\n" . $fix . ($c x $x) . $rest;
				&generate(($fix . ($choice x $x) . $rest), $result);
				push (@{$result},@temp );
			}
		return;
		}
	}
	else
	{
		my ($fix, $choice, $rest);

		($fix, $choice, $rest) = split( /\(([ACGT\/]+)\)(.*)$/i, $pattern);
		if (defined($fix) && defined($choice) && defined($rest)) 
		{ 
##			print "\nfix $fix, choice $choice, rest $rest"; 
			my (@temp,$i, $j, @cases);
			&generate ($rest, \@temp);
			@cases = split(/\//, $choice);
##			print "\ncases: @cases";
			foreach $i (@cases)
			{
##				print "\ncase: $i";
				if (scalar(@temp) == 0) 
				{	
					push (@{$result}, $fix . $i);
				}
				else
				{
					foreach $j (@temp)
					{
						push (@{$result}, $fix . $i . $j);
##						print "\n$fix . $i . $j\n";
					}
				}
			}
		}
##		print "\n$fix .  . $rest\n";
		if (!defined($rest)) 
		{ 
			push (@{$result}, $fix);
		}
		return ;
	}

}

#----------------------------------------------------------------
sub motif2hash( $ $ \@ )
{
	my ($site1, $site2, $select_nucl) = @_;
	my %motif_hash = ();
	my @R=();
	my @S=();
	my @sites1=();
	my @sites2=();
	my $motif='';
	my	$selectiv5 = '';
	my	$selectiv3 = '';
	my	$temp_site1= '';
	my	$temp_site2= '';
	my $restrictionSite1= '';
	my $restrictionSite2= '';
	my @rev_site1 = ();
	my @rev_R = ();
	my @rev_site2 = ();
	my @rev_S = ();
	my $rev_enzyme_site1= '';
	my $rev_enzyme_site2= '';
	my $rev_restrictionSite1 = '';
	my $rev_restrictionSite2 = '';
	my @site_select1 = ();
	my @rev_site_select1 = ();
	my @site_select2 = ();
	my @rev_site_select2 = ();
	
	my $IUPACsite1=&makeIUPAC($site1);
	my $IUPACsite2=&makeIUPAC($site2);
	
	&generate($IUPACsite1, \@R);
	@sites1 = map ($_, @R);
	&generate($IUPACsite2, \@S);
	@sites2 = map ($_, @S);
#=	foreach my $candidate_site (@sites1)
#=	{
#=		push(@rev_site1,&reverseComplement($candidate_site));
#=	}
#=	foreach my $candidate_site (@sites2)
#=	{
#=		push(@rev_site2,&reverseComplement($candidate_site));
#=	}
#=	push(@sites1,@rev_site1);
#=	push(@sites2,@rev_site2);
	
	my @tmp_enzyme1 = ();
	my @enzyme1 = ();
	
	foreach $motif (@sites1)
	{
		foreach my $selectivNucletide (@{$select_nucl})
		{
			print "selectiv5 $selectivNucletide->{'selectiv5'}\tselectiv3 $selectivNucletide->{'selectiv3'}\n";

			if($selectivNucletide->{'selectiv5'} ne 'N')			## use N if no selective nucleotide
			{
				$selectiv5 = $selectivNucletide->{'selectiv5'};
			}
			else
			{
				$selectiv5 = '';
			}	

			#-------------------------------
			#	with selective nucleotides
			#-------------------------------
			@tmp_enzyme1 = ();
			@enzyme1 = ();
			# -------FORWARD------
			$temp_site1= $motif . $selectiv5;
			$restrictionSite1=&makeIUPAC($temp_site1);

			&generate($restrictionSite1, \@R);
			@site_select1 = map ($_, @R);
			# -------REVERSE------
			@rev_site1 = ();
			@rev_R = ();
			$rev_enzyme_site1= &reverseComplement($temp_site1);
			$rev_restrictionSite1 = &makeIUPAC($rev_enzyme_site1);
			&generate($rev_restrictionSite1, \@rev_R);
			@rev_site_select1 = map ($_, @rev_R);
			# -----FORWARD & REVERSE----------
			@tmp_enzyme1 = (@tmp_enzyme1,@site_select1,@rev_site_select1);
	#-		@enzyme1 = &MakeNR(@tmp_enzyme1);
			$motif_hash{$site1}{$motif}=[@tmp_enzyme1];
		}
	}

	my @tmp_enzyme2 = ();
	my @enzyme2 = ();

	foreach $motif (@sites2)
	{
		foreach my $selectivNucletide (@{$select_nucl})
		{
			if($selectivNucletide->{'selectiv3'} ne 'N')			## use N if no selective nucleotide
			{
				$selectiv3 = $selectivNucletide->{'selectiv3'};
			}
			else
			{
				$selectiv3 = '';
			}	
			#-------------------------------
			#	with selective nucleotides
			#-------------------------------
			@tmp_enzyme2 = ();
			@enzyme2 = ();
			# -------FORWARD------
			$temp_site2= $motif . $selectiv3;
			$restrictionSite2=&makeIUPAC($temp_site2);

			&generate($restrictionSite2, \@S);
			@site_select2 = map ($_, @S);
			# -------REVERSE------
			@rev_site2 = ();
			@rev_S = ();
			$rev_enzyme_site2= &reverseComplement($temp_site2);
			$rev_restrictionSite2 = &makeIUPAC($rev_enzyme_site2);
			&generate($rev_restrictionSite2, \@rev_S);
			@rev_site_select2 = map ($_, @rev_S);
			# -----FORWARD & REVERSE----------
			@tmp_enzyme2 = (@tmp_enzyme2, @site_select2, @rev_site_select2);
	#-		@enzyme2 = &MakeNR(@tmp_enzyme2);
			$motif_hash{$site2}{$motif}=[@tmp_enzyme2];
		}
	}
	return(%motif_hash);
}
#------------------------------------------------------------
		##-------------------------------------------------
		## for Transcript profiling
		##-------------------------------------------------
sub transcript_profiling( \% $ $ $ \% )
{		
	my ($sequence, $adaptorLength, $real_site1, $real_site2, $motifs_hash) = @_;
	if ($sequence eq "" ) {return; }
	my $restrict1 = 0;
	my $restrict2 = 0;
	my $numSite1 = 0;
	my $numSite2 = 0;
	my $i = '';
	my $L_site2 = 0;
	my @TPF = ();
	my @TPF_temp = ();
	my $num_tpf = 0;
	my %sequence_file = ();
	my $first_cut = 0;
	
	foreach my $name (keys(%{$sequence}))
	{
		# ---- RUN RESTRICTION ANALYSIS 1 -----------------
		$numSite1 = 0;
		$first_cut = 0;
		my $hash_motif = $$motifs_hash{$real_site1};
		foreach my $i (keys(%$hash_motif))
		{
			$restrict1 = rindex($$sequence{$name}, $i);
			$first_cut = $restrict1 if($restrict1>0 && $first_cut<$restrict1);
		}		
		if ($first_cut>0 && $first_cut<length($$sequence{$name}))
		{
			# create subsequence from first restiction site on
			%sequence_file = ();
			my $sub_seq = uc(substr($$sequence{$name},$first_cut));
			$sequence_file{$name}=$sub_seq;
			# run AFLP on subsequence	
			@TPF_temp = &AmplifiedFragmentLenghtPolymorphism(\%sequence_file, $real_site1, $real_site2, \%{$motifs_hash});
				
			push(@TPF,@TPF_temp);
		}
	} # foreach $name (keys(%$sequence))
	return(@TPF);
}
#========================================================================================
		##-------------------------------------------------
		## for Amplified Fragment Length Polymorfisme
		##-------------------------------------------------
sub AmplifiedFragmentLenghtPolymorphism( \% $ $ \% )
{		
	my ($sequence, $real_site1, $real_site2, $motifs_hash) = @_;
	if ($sequence eq "" ) {return; }
	my @AFLP = ();
#	my @AFLP_temp =();
	my @AFLPsub =();
	my $num_fragments = 0;
	my @AFLP1 = ();
	my @AFLP2 = ();
#	my $numSite1 = 0;
	my $numSite2 = 0;
	my $restrict1 = 0;
	my $restrict2 = 0;
	my $i='';
	my $x=0 ;
	my $sequence_comment='';
	my @enzyme2='';
	my $rev_enzyme_site1='';
	my $rev_enzyme_site2='';
	my @rev_site1 = ();
	my @rev_site2 = ();
	my $selected_restrict1='';
	my $diff = 0;
	my $motif1 = '';
	my $motif2 = '';
	my $fragment = ''; 
	
	my $hash_motif1 = $$motifs_hash{$real_site1};
	my $hash_motif2 = $$motifs_hash{$real_site2};
	my %hash_motif = ();
	 %hash_motif = (%$hash_motif1,%$hash_motif2);
	
	foreach my $name (keys(%$sequence))
	{
		my @AFLP_temp =();
		my $numSite1 = 0;
		$sequence_comment	.= "$name\n";
		# ---- RUN RESTRICTION ANALYSES -----------------
		foreach my $motif (keys(%hash_motif))
		{
		#	foreach my $motif (@{$hash_motif{$key}})
		#	{
				$restrict1 = -1;
				do
				{	
					$restrict1 = index($$sequence{$name},$motif,($restrict1+1));
					if ($restrict1 > -1)
					{	
						$diff = length($hash_motif{$motif}[0])-length($motif);
						if($restrict1==0)
						{
							$selected_restrict1 = substr($$sequence{$name},$restrict1,length($hash_motif{$motif}[0])+$diff);
						}
						else
						{
							$selected_restrict1 = substr($$sequence{$name},$restrict1-$diff,length($hash_motif{$motif}[0])+$diff);
						}
					
						$AFLP_temp[$numSite1]{'enzyme'} = $motif;
						$AFLP_temp[$numSite1]{'site'} = $selected_restrict1;
						$AFLP_temp[$numSite1]{'pos'} = $restrict1;
						$numSite1++;
				#		last;
					}
				}
				until($restrict1 == -1);
		#	}
		}

		@AFLPsub =();
		@AFLPsub = sort {$a->{'pos'} <=> $b->{'pos'}} @AFLP_temp;

		# ---- RUN SELECTION STEP ---------------------
	#	$num_fragments = 0;
	#	print "coucou";
		my $max=scalar(@AFLPsub);
	 	for(my $x=0 ; $x<$max; $x++)
	 	{
			my @match = grep(m/$AFLPsub[$x]->{'enzyme'}/,(keys(%$hash_motif1)));
			if(scalar(@match)>0)
			{
				if($AFLPsub[$x-1]->{'enzyme'} ne $AFLPsub[$x]->{'enzyme'} && $x-1>=0)	# example: TTAAa~~~~~~~~~~~~~~~aGGWWCC
				{
					$motif1=$hash_motif{$AFLPsub[$x]->{'enzyme'}}[1];
					if(index($AFLPsub[$x]->{'site'},$motif1)>-1 && defined($motif1))
					{
							$motif2=$hash_motif{$AFLPsub[$x-1]->{'enzyme'}}[0];
							if(index($AFLPsub[$x-1]->{'site'},$motif2)>-1 && defined($motif2))		
							{
								$fragment = uc(substr($$sequence{$name}, $AFLPsub[$x-1]->{'pos'}, $AFLPsub[$x]->{'pos'}-$AFLPsub[$x-1]->{'pos'}+length($AFLPsub[$x]->{'enzyme'})));
						 		$AFLP[$num_fragments]{'name'} = ${name};
						 		$AFLP[$num_fragments]{'seqLength'} = (length($$sequence{$name})+1);
						 		$AFLP[$num_fragments]{'site1'} = $motif1;
						 		$AFLP[$num_fragments]{'pos1'} = $AFLPsub[$x-1]->{'pos'};
						 		$AFLP[$num_fragments]{'site2'} = $motif2;
						 		$AFLP[$num_fragments]{'pos2'} = $AFLPsub[$x]->{'pos'};
						 		$AFLP[$num_fragments]{'fragLength'} = length($fragment);
						 		$AFLP[$num_fragments]{'fragment'} = $fragment;
						 		$num_fragments++;$fragment = '';
				 			}
					}
				}
				if(defined($AFLPsub[$x+1]->{'enzyme'}) && $AFLPsub[$x]->{'enzyme'} ne $AFLPsub[$x+1]->{'enzyme'} && $x+1<=scalar(@AFLPsub))	# example: GGWWCCt~~~~~~~~~tTTAA
				{
					$motif1=$hash_motif{$AFLPsub[$x]->{'enzyme'}}[0];
					if(index($AFLPsub[$x]->{'site'},$motif1)>-1 && defined($motif1))
					{
							$motif2=$hash_motif{$AFLPsub[$x+1]->{'enzyme'}}[1];
							if(index($AFLPsub[$x+1]->{'site'},$motif2)>-1 && defined($motif2))		
							{
								$fragment = uc(substr($$sequence{$name}, $AFLPsub[$x]->{'pos'}, $AFLPsub[$x+1]->{'pos'}-$AFLPsub[$x]->{'pos'}+length($AFLPsub[$x+1]->{'enzyme'})));
						 		$AFLP[$num_fragments]{'name'} = ${name};
						 		$AFLP[$num_fragments]{'seqLength'} = (length($$sequence{$name})+1);
						 		$AFLP[$num_fragments]{'site1'} = $motif1;
						 		$AFLP[$num_fragments]{'pos1'} = $AFLPsub[$x]->{'pos'};
						 		$AFLP[$num_fragments]{'site2'} = $motif2;
						 		$AFLP[$num_fragments]{'pos2'} = $AFLPsub[$x+1]->{'pos'};
						 		$AFLP[$num_fragments]{'fragLength'} = length($fragment);
						 		$AFLP[$num_fragments]{'fragment'} = $fragment;
						 		$num_fragments++;$fragment = '';
							}
			 		}
		 		}
			}
		} #for($x=0 ; $x<scalar(@AFLPsub); $x++)
	} #foreach $name (keys(%$sequence))
	return(@AFLP);
}
	
#========================================================================================
		##-------------------------------------------------
		## for Amplified Fragment Length Polymorfisme
		##-------------------------------------------------
sub AmplifiedROELFragmentPolymorphism( \% $ $ \% )
{		
	my ($sequence, $real_site1, $real_site2, $motifs_hash) = @_;
	if ($sequence eq "" ) {return; }
	my @AFLP = ();
	my @AFLP_temp =();
	my @AFLPsub =();
	my $num_fragments = 0;
	my @AFLP1 = ();
	my @AFLP2 = ();
	my $numSite1 = 0;
	my $numSite2 = 0;
	my $restrict1 = 0;
	my $restrict2 = 0;
	my $i='';
	my $x=0 ;
	my $sequence_comment='';
	my @enzyme2='';
	my $rev_enzyme_site1='';
	my $rev_enzyme_site2='';
	my @rev_site1 = ();
	my @rev_site2 = ();
	my $selected_restrict1='';
	my $diff = 0;
	my $motif1 = '';
	my $motif2 = '';
	my $fragment = ''; 
	my $offset=0;
	my @restriction_list=();
	
	my $hash_motif1 = $$motifs_hash{$real_site1};
	my $hash_motif2 = $$motifs_hash{$real_site2};
	my %hash_motif = ();
	 %hash_motif = (%$hash_motif1,%$hash_motif2);
	
	foreach my $name (keys(%$sequence))
	{
		$sequence_comment	.= "$name\n";
		# ---- RUN RESTRICTION ANALYSES -----------------
		# for each motif check if rev-compl is the same motif or not (will not if not pallendromic)
		# the motifs are searched sequentially, which means that there is no way to avoid overlapping motifs at this step
		
		foreach my $key (keys(%hash_motif))
		{
			@restriction_list = ($key);
			push(@restriction_list,&reverseComplement($key)) if($key ne &reverseComplement($key));
			foreach my $motif (@restriction_list)
			{
				$restrict1 = -1 * length($motif);
				$offset=length($motif);
				do
				{	
					$restrict1 = index($$sequence{$name},$motif,($restrict1+$offset));
					if ($restrict1 > -1)
					{	
						$diff = length($hash_motif{$key}[0])-length($motif);
						if($restrict1==0)
						{
							$selected_restrict1 = substr($$sequence{$name},$restrict1,length($hash_motif{$key}[0])+$diff);
						}
						else
						{
							$selected_restrict1 = substr($$sequence{$name},$restrict1-$diff,length($hash_motif{$key}[0])+$diff);
						}
					
						$AFLP_temp[$numSite1]{'enzyme'} = $key;
						$AFLP_temp[$numSite1]{'site'} = $selected_restrict1;
						$AFLP_temp[$numSite1]{'pos'} = $restrict1;
						$numSite1++;
				#		last;
					}
				}
				until($restrict1 == -1);
			}
		}

		@AFLPsub =();
		@AFLPsub = sort {$a->{'pos'} <=> $b->{'pos'}} @AFLP_temp;

		# ---- RUN SELECTION STEP ---------------------
		# overlapping motifs have to be taken into account here by shifting one position in the array away from the site of interest
		my $z=0;
		while(defined($AFLPsub[$z]->{'enzyme'}))
		{
			if(abs($AFLPsub[$z]->{'pos'} - $AFLPsub[$z+1]->{'pos'})<length($AFLPsub[$z]->{'enzyme'}))
			{
				if(length($AFLPsub[$z]->{'enzyme'})>=length($AFLPsub[$z+1]->{'enzyme'}))
				{
					splice(@AFLPsub,($z+1),1);					
					$z--;
				}
				else
				{
					splice(@AFLPsub,$z,1);
					$z--;
				}
			}
			$z++;
		}
		$num_fragments = 0;
	#	print "coucou";
		my $max=scalar(@AFLPsub);
		
	 	for(my $x=0 ; $x<$max; $x++)
	 	{
			my @match = grep(m/$AFLPsub[$x]->{'enzyme'}/,(keys(%$hash_motif1)));
			if(scalar(@match)>0)
			{
			  if($x>1)
			  {
				  if($AFLPsub[$x-1]->{'enzyme'} ne $AFLPsub[$x]->{'enzyme'})	# example: TTAAa~~~~~~~~~~~~~~~aGGWWCC
				  {
						$motif1=$hash_motif{$AFLPsub[$x]->{'enzyme'}}[1];
						if(index($AFLPsub[$x]->{'site'},$motif1)>-1 && defined($motif1))
						{
							$motif2=$hash_motif{$AFLPsub[$x-1]->{'enzyme'}}[0];
							if(index($AFLPsub[$x-1]->{'site'},$motif2)>-1 && defined($motif2))		
							{
								$fragment = uc(substr($$sequence{$name}, $AFLPsub[$x-1]->{'pos'}, $AFLPsub[$x]->{'pos'}-$AFLPsub[$x-1]->{'pos'}+length($AFLPsub[$x]->{'enzyme'})));
						 		$AFLP[$num_fragments]{'name'} = ${name};
						 		$AFLP[$num_fragments]{'seqLength'} = (length($$sequence{$name})+1);
						 		$AFLP[$num_fragments]{'site1'} = $motif1;
						 		$AFLP[$num_fragments]{'pos1'} = $AFLPsub[$x-1]->{'pos'};
						 		$AFLP[$num_fragments]{'site2'} = $motif2;
						 		$AFLP[$num_fragments]{'pos2'} = $AFLPsub[$x]->{'pos'};
						 		$AFLP[$num_fragments]{'fragLength'} = length($fragment);
						 		$AFLP[$num_fragments]{'fragment'} = $fragment;
						 		$num_fragments++;$fragment = '';
				 			}
						}
				  }
				}			
				if($x+1<=scalar(@AFLPsub))
				{
				  if($AFLPsub[$x]->{'enzyme'} ne $AFLPsub[$x+1]->{'enzyme'})	# example: GGWWCCt~~~~~~~~~tTTAA
				  {
						$motif1=$hash_motif{$AFLPsub[$x]->{'enzyme'}}[0];
						if(index($AFLPsub[$x]->{'site'},$motif1)>-1 && defined($motif1))
						{
							$motif2=$hash_motif{$AFLPsub[$x+1]->{'enzyme'}}[1];
							if(index($AFLPsub[$x+1]->{'site'},$motif2)>-1 && defined($motif2))		
							{
								$fragment = uc(substr($$sequence{$name}, $AFLPsub[$x]->{'pos'}, $AFLPsub[$x+1]->{'pos'}-$AFLPsub[$x]->{'pos'}+length($AFLPsub[$x+1]->{'enzyme'})));
						 		$AFLP[$num_fragments]{'name'} = ${name};
						 		$AFLP[$num_fragments]{'seqLength'} = (length($$sequence{$name})+1);
						 		$AFLP[$num_fragments]{'site1'} = $motif1;
						 		$AFLP[$num_fragments]{'pos1'} = $AFLPsub[$x]->{'pos'};
						 		$AFLP[$num_fragments]{'site2'} = $motif2;
						 		$AFLP[$num_fragments]{'pos2'} = $AFLPsub[$x+1]->{'pos'};
						 		$AFLP[$num_fragments]{'fragLength'} = length($fragment);
						 		$AFLP[$num_fragments]{'fragment'} = $fragment;
						 		$num_fragments++;$fragment = '';
							}
						}
			 		}
		 		}
			}
		} #for($x=0 ; $x<scalar(@AFLPsub); $x++)
	} #foreach $name (keys(%$sequence))
	return(@AFLP);
}
	
#========================================================================================
		##-------------------------------------------------
		## for statistics on one site ... use e.g. AFLPinSilico.pl files CCWWGG N 0 N N -site
		##-------------------------------------------------
sub sitestat( \% $ )
{		
	my ($sequence, $real_site1) = @_;
	if ($sequence eq "" ) {return; }
	my $i = 0;
	my @AFLP1 = ();
	my $numSite1 = 0;
	my $restrict1 = 0;
	my @site1 = ();
	my @rev_site1 = ();
	my @site_stat = ();
	my @R = ();
	my @rev_R = ();
	my $restrictionSite1 = &makeIUPAC($real_site1);
	my $rev_real_site1= &reverseComplement($real_site1);
	my $rev_restrictionSite1 = &makeIUPAC($rev_real_site1);

	&generate($restrictionSite1, \@R);
	@site1 = map ($_, @R);
	&generate($rev_restrictionSite1, \@rev_R);
	@rev_site1 = map ($_, @rev_R);
	
	my @tmp_enzyme1 = (@site1,@rev_site1);
	my @enzyme = &MakeNR(@tmp_enzyme1);

	foreach my $name (keys(%$sequence))
	{
		$numSite1 = 0;
		@AFLP1 = ();
		foreach $i (@enzyme)
		{
			$restrict1 = 0;
			while(index($$sequence{$name}, $i,($restrict1+1))>-1)
			{
				if ($restrict1<index($$sequence{$name}, $i,($restrict1+1)))
				{
					$restrict1 = index($$sequence{$name}, $i,($restrict1+1));
					$AFLP1[$numSite1]{'enzyme'} = $real_site1;
					$AFLP1[$numSite1]{'site'} = $i;
					$AFLP1[$numSite1]{'pos'} = $restrict1;
				}
				$numSite1++;
			}
		}

		my $max_restrict = 0;
		foreach my $x (@AFLP1)
		{
			$max_restrict = $x->{'pos'};
		} 
		push(@site_stat, "$name\t" . scalar(@AFLP1) . "\t" . $max_restrict .  "\t" . (length($$sequence{$name})-$max_restrict) . "\t\t" . length($$sequence{$name}) . "\t" . (length($$sequence{$name})/scalar(@AFLP1)) . "\n");
	}
	return(@site_stat);
}
#========================================================================================
		##-------------------------------------------------
		## for restriction digest ... use e.g. AFLPinSilico.pl files CCWWGG N 0 N N -site
		##-------------------------------------------------
sub Digest( \% $ $ )
{		
	my ($sequence, $real_site1, $site) = @_;
	if ($sequence eq "" ) {return; }
	my $restrictionSite1 = &makeIUPAC($real_site1);
	my $rev_real_site1= &reverseComplement($real_site1);
	my $rev_restrictionSite1 = &makeIUPAC($rev_real_site1);
	my @R = ();
	my @rev_R = ();
	my @DIGEST = ();
	
	&generate($restrictionSite1, \@R);
	my @site1 = map ($_, @R);
	&generate($rev_restrictionSite1, \@rev_R);
	my @rev_site1 = map ($_, @rev_R);
	
	my @tmp_enzyme1 = (@site1,@rev_site1);
	my @enzyme = &MakeNR(@tmp_enzyme1);

	foreach my $name (keys(%$sequence))
	{
		my $seq = $$sequence{$name};
		my @SITES = (0,length($seq));
		foreach my $i (@enzyme)
		{
			my $from = 0;
			my $restrict = index($seq, $i);
			while($restrict>-1)
			{
				push(@SITES,$restrict);
				$from = $restrict;
				$restrict = index($seq, $i,($from+1));
			}
		}
		if(scalar(@SITES)>2)
		{
			@SITES = sort { $a <=> $b } @SITES;
			for(my $x=0 ; scalar(@SITES)>($x+1) ; $x++)
			{
				my %tmp = ();
				$tmp{'source'} = $name;
				$tmp{'enzyme'} = $real_site1;
				$tmp{'pos'} = ($SITES[$x]+1) .'..'. ($SITES[($x+1)]+length($real_site1));
				$tmp{'frag'} = substr($seq,$SITES[$x],($SITES[($x+1)]+length($real_site1)-$SITES[$x]));
				foreach my $i (@site1)
				{
					$tmp{'frag'} =~ s/$i/$site/g;
				}
				foreach my $i (@rev_site1)
				{
					my $rev_site = &reverseComplement($site);
					$tmp{'frag'} =~ s/$i/$rev_site/g;
				}
				push(@DIGEST,\%tmp);
			}
		}
	}
	return(@DIGEST);
}
#========================================================================================
#reduces redundancy in array
sub MakeNR( @ )
{
	my (@enzyme) = @_;
	my $x=0 ;
	@enzyme = sort(@enzyme);
	while(defined($enzyme[$x+1]))
	{
		if($enzyme[$x] eq $enzyme[$x+1])
		{
			splice(@enzyme,($x+1),1);
		}
		else
		{
			$x++;
		}
	}
	return(@enzyme);
}
#============================================================================
#============================================================================


1;

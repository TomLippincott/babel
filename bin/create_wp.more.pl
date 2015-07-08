use utf8;
use Encode;
use open ':encoding(utf8)';

binmode(STDIN, ":utf8");
binmode(STDOUT, ":utf8");
binmode(STDERR, ":utf8");

die ("DICT_IV DICT_OOV DICT_OOVextra data/OFST transparent") if ($#ARGV<4);
($dict_iv,$dict_oov,$dict_oov_extra,$outdir,$transparent)=@ARGV;

system("mkdir -p $outdir");

$words2phones_fsm=$outdir."/words2phones.fsm";
open(f_w2p,">$words2phones_fsm");
$words2phones_fst=$outdir."/words2phones.fst";


$words_file=$outdir."/words.sym";
open(f_w,">$words_file");
$phones_file=$outdir."/phones.sym";
open(f_p,">$phones_file");

$fst_header=$outdir."/fst_header";
open(f_h,">$fst_header");
print f_h "# input-voc: /dynamic/\n";
print f_h "# output-voc: /dynamic/\n";
close(f_h);

$fst_header=$outdir."/fsa_header";
open(f_h,">$fsa_header");
print f_h "# input-voc: /dynamic/\n";
close(f_h);


if ($dict_iv=~/\.gz$/){
  open(f_iv,"gunzip -c $dict_iv |") ||  die "I can't open $dict_iv";
}
else{
  open(f_iv,$dict_iv) ||  die "I can't open $dict_iv";
}

if ($dict_oov=~/\.gz$/){
  open(f_oov,"gunzip -c $dict_oov |") ||  die "I can't open $dict_oov";
}
else{
  open(f_oov,$dict_oov) ||  die "I can't open $dict_oov";
}

if ($dict_oov_extra=~/\.gz$/){
  open(f_oov,"gunzip -c $dict_oov_extra |") ||  die "I can't open $dict_oov_extra";
}
else{
  open(f_oov_extra,$dict_oov_extra) ||  die "I can't open $dict_oov_extra";
}


print f_w "<epsilon> 0\n";
print f_p "<epsilon> 0\n";


$count=1;
while(<f_iv>){
  chop;
  $_=~s/\[ wb \]//g;
  $_=~s/\[wb\]//g;
  undef @s;
  @s=split(/\s+/);
  @b=split(/\(/,$s[0]);
  $word=$b[0];
  if (&is_transparent($word,$transparent)){
    for($i=1;$i<=$#s;$i++){ $p{$s[$i]}=1; };
  }
  else{
    $w{$word}=1;
    $p{$s[1]}=1;
    
    print f_w2p "0 ",$count," ",$word," ",$s[1],"\n";
    
    for($i=2;$i<=$#s;$i++){
      if ($s[$i]=~/\[/){
      }
      else{
	$p{$s[$i]}=1;
	print f_w2p $count," ",$count+1," <epsilon> ",$s[$i],"\n";
	$count++;
      }
    }
    $final{$count}=1;  
    $count++;
  }
}
close(f_iv);

while(<f_oov>){
  chop;
  $_=~s/\[ wb \]//g;
  $_=~s/\[wb\]//g;
  undef @s;
  @s=split(/\s+/);
  @b=split(/\(/,$s[0]);
  $word=$b[0];
  
  if (! &is_transparent($word,$transparent)){
    $w{$word}=1;
    $p{$s[1]}=1;
    print f_w2p "0 ",$count," ",$word," ",$s[1],"\n";
    for($i=2;$i<=$#s;$i++){
      if ($s[$i]=~/\[/){
      }
      else{
	$p{$s[$i]}=1;
	print f_w2p $count," ",$count+1," <epsilon> ",$s[$i],"\n";
	$count++;
      }
    }
    shift(@s); $jj=join(" ",@s); $is{$word}{$jj}=1; $isoov{$word}=1;
    $final{$count}=1;
    $count++;
  }
}
close(f_oov);

while(<f_oov_extra>){
  chop;
  $_=~s/\[ wb \]//g;
  $_=~s/\[wb\]//g;
  undef @s;
  @s=split(/\s+/);
  @b=split(/\(/,$s[0]);
  $word=$b[0];
  shift(@s); $jj=join(" ",@s); 
  #print "WORD $word  $isoov{$word}\n";
  #print "PRON $jj  $is{$word}{$jj}\n";
  if ($isoov{$word} && ($is{$word}{$jj} eq "")){
    $p{$s[0]}=1;
    print f_w2p "0 ",$count," ",$word," ",$s[0],"\n";
    for($i=1;$i<=$#s;$i++){
      if ($s[$i]=~/\[/){
      }
      else{
        $p{$s[$i]}=1;
        print f_w2p $count," ",$count+1," <epsilon> ",$s[$i],"\n";
        $count++;
      }
    }
    $final{$count}=1;
    $count++;
  }
}
close(f_oov_extra);


while(($k,$v)=each %final){
  print f_w2p $k," 0 <epsilon> <epsilon>\n";
}
print f_w2p "0\n";

close(f_w2p);


$count=1;
while(($k,$v)=each %w){
  print f_w $k," ",$count,"\n";
  $count++;
}
$count=1;

while(($k,$v)=each %p){
  print f_p $k," ",$count,"\n";
  $count++;
}
close(f_w);
close(f_p);

$cmd1="fstcompile -isymbols=$words_file -osymbols=$phones_file $words2phones_fsm | fstarcsort --sort_type=ilabel - $words2phones_fst";

syscmd($cmd1);


sub syscmd {
    my ($cmd)=@_;
    print STDERR "syscmd: $cmd\n";
    my $rc=system($cmd);
    print STDERR "rc=$rc\n";
    if ($rc){
	print STDERR "(ERROR) System call:\n";
	print STDERR "$cmd\n";
	print STDERR "\treturns rc=$rc\n";
	die;
    }
}
 
sub is_transparent{
 my($w,$t)=@_;
 my @a=split(/\,/,$t);
 foreach (@a){
   if ($w eq $_){
     return 1;
   }
 }
 return 0;
}




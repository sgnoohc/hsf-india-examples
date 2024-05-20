"  .
" ..: P. Chang,  philip@physics.ucsd.edu


"NeoBundle Scripts-----------------------------
if &compatible
  set nocompatible               " Be iMproved
endif
" Required:
set runtimepath+=~/.vim/bundle/neobundle.vim/
" Required:
call neobundle#begin(expand('~/.vim/bundle'))
" Let NeoBundle manage NeoBundle
" Required:
" NeoBundleFetch 'Shougo/neobundle.vim'
" Add or remove your Bundles here:
" NeoBundle 'Shougo/neosnippet.vim'
" NeoBundle 'Shougo/neosnippet-snippets'
" NeoBundle 'tpope/vim-fugitive'
" NeoBundle 'ctrlpvim/ctrlp.vim'
" NeoBundle 'flazz/vim-colorschemes'
" You can specify revision/branch/tag.
" NeoBundle 'Shougo/vimshell', { 'rev' : '3787e5' }
" NeoBundle 'majutsushi/tagbar'
" NeoBundle 'ap/vim-buftabline'
" NeoBundle 'vim-airline/vim-airline'
" NeoBundle 'vim-airline/vim-airline-themes'
" NeoBundle 'vim-scripts/a.vim'
" NeoBundle 'vim-scripts/vis'
" NeoBundle 'junegunn/vim-easy-align'
" NeoBundle 'Align'
" NeoBundle 'roman/golden-ratio'
"NeoBundle 'soramugi/auto-ctags.vim'
" NeoBundle 'triglav/vim-visual-increment'
" NeoBundle 'haya14busa/incsearch.vim'
" NeoBundle 'haya14busa/incsearch-fuzzy.vim'
" NeoBundle 'tpope/vim-commentary'
" NeoBundle 'bling/vim-bufferline'
" NeoBundle 'airblade/vim-gitgutter'
" NeoBundle 'tmhedberg/matchit'
"NeoBundle 'scrooloose/nerdtree'
" NeoBundle 'jiangmiao/auto-pairs'
" NeoBundle 'tpope/vim-surround'
" NeoBundle 'aminnj/vim-lazytools'
" NeoBundle 'google/vim-searchindex'
" NeoBundle 'JuliaEditorSupport/julia-vim'
" Required:
call neobundle#end()
" Required:
filetype plugin indent on
" If there are uninstalled bundles found on startup,
" this will conveniently prompt you to install them.
NeoBundleCheck
"End NeoBundle Scripts-------------------------


"source $HOME/login/vim/a.vim
"source $HOME/login/vim/minibufexpl.vim
"source $HOME/login/vim/vim-geeknote-master/plugin/vim_geeknote.vim
"source $HOME/login/vim/taglist.vim
"source $HOME/login/.vim/plugin/airline.vim
filetype plugin on

"==================="
" Personal Settings "
"==================="

"escape with jk
:imap jk <Esc>
syntax on                       "turning on syntax coloring
set tabstop=8
set softtabstop=4
set shiftwidth=4
set expandtab
set cmdheight=2                 " Following lets me not having to Press ENTER like a dog
set ruler                       " Ruler to be shown on the bottom right corner
set hlsearch
set wildmenu
set number
" keep cursor in the middle
set scrolloff=40
" highlight current line
"set cursorline
set nocursorcolumn
set nowrap
" tabs are replaced with chracters
set list
set listchars=tab:>-
let mapleader=","

"
vnoremap . :norm .<cr>

" For python
autocmd FileType python set tabstop=4|set shiftwidth=4|set expandtab

hi CursorLine   cterm=NONE ctermbg=black guibg=darkred guifg=white
"nnoremap <Leader>c :set cursorline! cursorcolumn!<CR>
nnoremap <Leader>c :set cursorline! <CR>
nnoremap <Leader><Leader> :wqa!<CR>
nnoremap <Leader>] :q<CR>

set pastetoggle=<C-Y>

"set comments=sO:*\ -,mO:*\ \ ,exO:*/,s1:/*,mb:*,ex:*/,bO:///,O://
set ic
autocmd FileType * setlocal formatoptions-=c formatoptions-=r formatoptions-=o
set nocin noai nosi inde=
let thisos=$THISOS
if thisos == "Windows"
    set backspace=2
    set backspace=indent,eol,start
    set backupdir=~/.vim/backup//
    set directory=~/.vim/swap//
    set undodir=~/.vim/undo//
endif
autocmd BufNewFile,BufRead *.v e ++ff=dos | set tabstop=3 | set syntax=verilog
autocmd BufNewFile,BufRead *.vh e ++ff=dos | set tabstop=3 | set syntax=verilog
autocmd BufNewFile,BufRead *.vhd e ++ff=dos | set tabstop=3 | set syntax=verilog
autocmd BufNewFile,BufRead *.def set syntax=cfg
autocmd BufNewFile,BufRead *.cc_ set syntax=c
autocmd BufNewFile,BufRead *.cu set syntax=c
autocmd BufNewFile,BufRead *.cuh set syntax=c
autocmd BufNewFile,BufRead *.tex set wrap
autocmd BufNewFile,BufRead *.md set virtualedit=all
au BufNewFile,BufFilePre,BufRead *.md set filetype=markdown
au BufNewFile,BufFilePre,BufRead *.markdown e ++enc=utf-8
au BufNewFile,BufFilePre,BufRead *.markdown set filetype=txt

nnoremap <Leader>rw :%s/\s\+$//e<CR>
vnoremap gw :s![^ ]\zs  \+! !g<CR>

"set formatoptions=

"set mouse=a                     "Set mouse active in iTerm (MacOSX?anyone?)
" set autoindent
"set smartindent
"set cursorline                  "Current cursorline option
"set textwidth=79                " break lines when line length increases

""======================="
"" trailing white spaces "
""======================="
"set list listchars=tab:>-,trail:.,extends:>
"" Enter the middle-dot by pressing Ctrl-k then .M
"set list listchars=tab:\|_,trail:·
"" Enter the right-angle-quote by pressing Ctrl-k then >>
"set list listchars=tab:»·,trail:·
"" Enter the Pilcrow mark by pressing Ctrl-k then PI
""set list listchars=tab:>-,eol:¶


"========="
" Mapping "
"========="

" Press F4 to toggle highlighting on/off, and show current value.
:noremap <F4> :set hlsearch! hlsearch?<CR>

" buffer
nmap <leader>s<left>   :leftabove  vnew<CR>
nmap <leader>s<right>  :rightbelow vnew<CR>
nmap <leader>s<up>     :leftabove  new<CR>
nmap <leader>s<down>   :rightbelow new<CR>
nmap <leader>t8        :set tabstop=8<CR>
nmap <leader>t4        :set tabstop=4<CR>
nmap <leader>t2        :set tabstop=2<CR>

" cxx <--> h
nmap <leader>d         :A<CR>

" Search
set incsearch       " shows matches halfway typing a pattern
set noignorecase          " (no)ignore case in searching
set hlsearch                " (no)highlight
" nnoremap <silent> <Space> :silent noh<Bar>echo<CR> " Spacebar to clear search
nnoremap <Space> %
vnoremap <Space> %
nnoremap <Leader><Space> :noh<CR>

" Setting a foldmethod to be initiated when I type \zs
nmap <leader>zs        :set foldmethod=syntax<CR>
set foldopen-=block

if bufwinnr(1)
  map + <C-W>+
  map - <C-W>-
endif

" Commenting/uncommenting like CTRL-CC for emacs
map ,/ :s/^/\/\//<CR><Space>  " c/c++ comment
map ,? :s/^\/\///<CR><Space>  " c/c++ uncomment
map ,# :s/^/#/<CR><Space>     " shell comment
map ,3 :s/^#//<CR><Space>     " shell uncomment
map ,% :s/^/%/<CR><Space>     " latex comment
map ,5 :s/^%//<CR><Space>     " latex uncomment
map ," :s/^/"/<CR><Space>     " vimrc comment
map ,' :s/^"//<CR><Space>     " vimrc uncomment

"noremap <Leader>j J
"noremap K 5<up>
"noremap J 5<down>

"========="
" Buffers "
"========="

" It's useful to show the buffer number in the status line.
"set laststatus=1 "statusline=%02n:%<%f\ %h%m%r%=%-14.(%l,%c%V%)\ %P

" Conveniently accessing buffers
"set wildchar=<Tab> wildmenu wildmode=full
"set wildcharm=<C-Z>
"nnoremap <F10> :b <C-Z>

" Mappings to access buffers (don't use "\p" because a
" delay before pressing "p" would accidentally paste).
" \l       : list buffers
" \b \f \g : go back/forward/last-used
" \1 \2 \3 : go to buffer 1/2/3 etc
nnoremap <Leader>` :ls<CR>
nnoremap <Leader>b :bp<CR>
nnoremap <Leader>f :bn<CR>
nnoremap <Leader>[ :bw<CR>
"nnoremap <Leader>d :bd<CR>
nnoremap <Leader>g :e#<CR>
nnoremap <Leader>1 :1b<CR>
nnoremap <Leader>2 :2b<CR>
nnoremap <Leader>3 :3b<CR>
nnoremap <Leader>4 :4b<CR>
nnoremap <Leader>5 :5b<CR>
nnoremap <Leader>6 :6b<CR>
nnoremap <Leader>7 :7b<CR>
nnoremap <Leader>8 :8b<CR>
nnoremap <Leader>9 :9b<CR>
nnoremap <Leader>00 :10b<CR>
nnoremap <Leader>01 :11b<CR>
nnoremap <Leader>02 :12b<CR>
nnoremap <Leader>03 :13b<CR>
nnoremap <Leader>04 :14b<CR>
nnoremap <Leader>05 :15b<CR>
nnoremap <Leader>06 :16b<CR>
nnoremap <Leader>07 :17b<CR>
nnoremap <Leader>08 :18b<CR>
nnoremap <Leader>09 :19b<CR>
nnoremap <Leader>90 :20b<CR>
nnoremap <Leader>91 :21b<CR>
nnoremap <Leader>92 :22b<CR>
nnoremap <Leader>93 :23b<CR>
nnoremap <Leader>94 :24b<CR>
nnoremap <Leader>95 :25b<CR>
nnoremap <Leader>96 :26b<CR>
nnoremap <Leader>97 :27b<CR>
nnoremap <Leader>98 :28b<CR>
nnoremap <Leader>99 :29b<CR>

" " Save folds when closed, load folds when opened.
" "au BufWinLeave * silent! mkview
" "au BufWinEnter * silent! loadview
" "
" "
" Tab completion of words in the document
function! InsertTabWrapper()
  let col = col('.') - 1
  if !col || getline('.')[col - 1] !~ '\k'
    return "\<tab>"
  else
    return "\<c-n>"
  endif
endfunction
inoremap <tab> <c-r>=InsertTabWrapper()<cr>

"set term=builtin_ansi

"======================"
" Environment settings "
"======================"

" Tell vim to remember certain things when we exit
"  '10  :  marks will be remembered for up to 10 previously edited files
"  "100 :  will save up to 100 lines for each register
"  :20  :  up to 20 lines of command-line history will be remembered
"  %    :  saves and restores the buffer list
"  n... :  where to save the viminfo files
set viminfo='10,\"100,:20,%,n~/.viminfo

" When editing a file, always jump to the last cursor position
 au BufReadPost *
       \ if ! exists("g:leave_my_cursor_position_alone") |
       \     if line("'\"") > 0 && line ("'\"") <= line("$") |
       \         exe "normal g'\"" |
       \     endif |
       \ endif


if filereadable(".vim.custom")
  so .vim.custom
endif

"let g:SuperTabDefaultCompletionType = "<c-p>"

highlight Pmenu ctermbg=black ctermfg=white gui=bold
highlight PmenuSel ctermbg=white ctermfg=black gui=bold

" You need to use ctermbg ctermfg instead of guibg, guifg
" http://vim.1045645.n5.nabble.com/Fold-color-td2845368.html
highlight Folded ctermbg=none ctermfg=DarkGreen
highlight FoldColumn ctermbg=darkgrey ctermfg=white
"au BufWinLeave * mkview
"au BufWinEnter * silent loadview
"set foldmethod=syntax
"set foldlevel=20
"set foldnestmax=1
"set fdo-=search
let c_no_comment_fold = 1 "http://stackoverflow.com/questions/10038477/vim-syntax-folding-disable-folding-multi-line-comments
inoremap <F9> <C-O>za
nnoremap <F9> za
onoremap <F9> <C-C>za
vnoremap <F9> zf

" stop from fold opening when an unmatched paranthese is given
" http://stackoverflow.com/questions/4630892/vim-folds-open-up-when-giving-an-unmatched-opening-brace-parenthesis
"autocmd InsertEnter * if !exists('w:last_fdm') | let w:last_fdm=&foldmethod | setlocal foldmethod=manual | endif
"autocmd InsertLeave,WinLeave * if exists('w:last_fdm') | let &l:foldmethod=w:last_fdm | unlet w:last_fdm | endif

" Do not create swap file for dropbox
"autocmd BufNewFile,BufRead *
"  \ if expand('%:p') =~ '/home/phchang/phchang/Dropbox' |
"  \   set noswapfile |
"  \ else |
"  \   set swapfile |
"  \ endif

" custom highlighting
highlight MyRed ctermfg=red
syntax match MyRed /ERROR/
syntax match MyRed /Error/
syntax match MyRed /error/
highlight MyYellow ctermfg=yellow
syntax match MyYellow /WARNING/
syntax match MyYellow /Warning/
syntax match MyYellow /warning/

" diff setting
highlight DiffAdd    cterm=none ctermbg=DarkGray
highlight DiffDelete cterm=none ctermbg=Black
highlight DiffChange cterm=none ctermbg=Black
highlight DiffText   cterm=none ctermfg=yellow ctermbg=Black

let c_no_curly_error=1

"https://superuser.com/questions/990296/how-to-change-the-way-that-vim-displays-collapsed-folded-lines
function! NeatFoldText()
    let line = ' ' . substitute(getline(v:foldstart), '^\s*"\?\s*\|\s*"\?\s*{{' . '{\d*\s*', '', 'g') . ' '
    let lines_count = v:foldend - v:foldstart + 1
    let lines_count_text = '| ' . printf("%10s", lines_count . ' lines') . ' |'
    let foldchar = matchstr(&fillchars, 'fold:\zs.')
    let foldtextstart = strpart('+' . repeat(foldchar, v:foldlevel*2) . line, 0, (winwidth(0)*2)/3)
    let foldtextend = lines_count_text . repeat(foldchar, 8)
    let foldtextlength = strlen(substitute(foldtextstart . foldtextend, '.', 'x', 'g')) + &foldcolumn
    return foldtextstart . repeat(foldchar, winwidth(0)-foldtextlength) . foldtextend
    "return foldtextstart
endfunction

set foldtext=NeatFoldText()

if exists('$CMSSW_BASE')
    " search CMS3 first, then local CMSSW, then central CMSSW
    set path+=$CMSSW_BASE/src/CMS3/NtupleMaker/src
    set path+=$CMSSW_BASE/src
    set path+=$CMSSW_RELEASE_BASE/src
    set path+=$ROOTSYS/include
    " remove includes from autocomplete search list, otherwise slow
    set complete-=i
endif

if exists('$ANALYSIS_BASE')
    " search CMS3 first, then local CMSSW, then central CMSSW
    set path+=$ANALYSIS_BASE
    " remove includes from autocomplete search list, otherwise slow
    set complete-=i
endif

" Airline vim settings
set laststatus=2
set t_Co=256
set cmdheight=1
let g:airline#extensions#tabline#enabled = 1
let g:airline#extensions#tabline#fnamemod = ':t'
let g:airline#extensions#tabline#buffer_idx_mode = 1
let g:airline#extensions#tabline#show_buffers = 1
let g:airline#extensions#tagbar#flags = "f"  " show full tag hierarchy
let g:airline#extensions#whitespace#max_lines = 100

let g:airline_theme='badwolf'
"let g:tagbar_ctags_bin='~/software/bin/ctags'

nmap <F8> :TagbarToggle<CR>
autocmd FileType * setlocal formatoptions-=c formatoptions-=r formatoptions-=o

"map <C-\> :Ctags<CR>:vsp <CR>:exec("tag ".expand("<cword>"))<CR>
map <C-\> :tab<CR>:vsp <CR>:exec("tag ".expand("<cword>"))<CR>

" set term=screen-256color

"color wombat256mod

"nmap <F6> :colorscheme wombat256mod<CR>
"nmap <F7> :colorscheme default<CR>

nnoremap <script> <F7> :call ChangeColorToDefault()<cr>
function! ChangeColorToDefault()
    :colorscheme default
    set cursorline!
    call writefile([':colorscheme default', 'set cursorline!'], expand('~/.vim/persisted_options.vim'))
endfunction

nnoremap <script> <F6> :call ChangeColorToWombat()<cr>
function! ChangeColorToWombat()
    :colorscheme wombat256mod
    set cursorline
    call writefile([':colorscheme wombat256mod', 'set cursorline'], expand('~/.vim/persisted_options.vim'))
endfunction


"map <C-c> :s/^/\/\//<Enter><Space>
"map <C-u> :s/^\/\///<Enter><Space>

"nnoremap <Leader>c, :s/,/,\ /g<CR>
"nnoremap <Leader>ci :s/if(/if\ (/g<CR>
"nnoremap <Leader>cf :s/for(/for\ (/g<CR>
"nnoremap <Leader>c/ :s/\//\ \/\ /g<CR>
"nnoremap <Leader>c+ :s/+/\ +\ /g<CR>
"nnoremap <Leader>c- :s/-/\ -\ /g<CR>
"nnoremap <Leader>c= :s/=/\ =\ /g<CR>
"nnoremap <Leader>c< :s/</\ <\ /g<CR>
"nnoremap <Leader>c; :s/;/\<\ /g<CR>
"nnoremap <Leader>c( :s/(\ /(/g<CR>
"nnoremap <Leader>c) :s/)\ /)/g<CR>
"nnoremap <Leader>c{ :s/){/)\ {/g<CR>

" Use artistic style
autocmd BufNewFile,BufRead *.cc,*.h,*.C,*.cxx set formatprg=clang-format\ -style=\"{BasedOnStyle:\ llvm,\ IndentWidth:\ 4,\ ColumnLimit:\ 1000,\ AllowShortIfStatementsOnASingleLine:\ true,\ AllowShortBlocksOnASingleLine:\ false,\ BreakBeforeBraces:\ Allman}\"

" Add expression under cursor in real time
nnoremap <leader>cc ciW<C-r>=<C-r>"<CR><Esc>

set tags=.tags

inoremap <C-d> <Del>

"nnoremap ,/ I//<Esc>  " c/c++ comment
"nnoremap ,? I<Del><Del><Esc>  " c/c++ uncomment

let g:alternateExtensions_h = "C,c,cpp,cxx,cc,CC,cu"
let g:alternateExtensions_H = ""
let g:alternateExtensions_cpp = "h,hpp"
let g:alternateExtensions_CPP = "h,hpp"
let g:alternateExtensions_c = "h,cuh"
let g:alternateExtensions_C = "h,cuh"
let g:alternateExtensions_cxx = "h,cuh"
let g:alternateExtensions_cu = "cuh"
let g:alternateExtensions_cuh = "cu"

" ctags
"let g:auto_ctags_tags_name = '.tags'

" https://github.com/junegunn/vim-easy-align
" Start interactive EasyAlign in visual mode (e.g. vipga)
xmap ga <Plug>(EasyAlign)

" Start interactive EasyAlign for a motion/text object (e.g. gaip)
nmap ga <Plug>(EasyAlign)

source $HOME/.vim/persisted_options.vim


autocmd FileType c,cpp,cuda nnoremap <leader>cp :call lazytools#CoutTokens()<CR>
"" https://github.com/Amarang/syncfiles/blob/082b4b9e1de144ec87480d0bfe52e9659d0a1748/dotfiles/vimrc#L428-L462
"fu! CoutTokens()
"    " toggles between
"    "    std::cout << " blah1: " << blah1 << " blah2: " << blah2 << " blah3: " << blah3 << std::endl;
"    " and
"    "    blah1 blah2 blah3  
"    "
"    let line=getline('.')
"    " turn into cout statement or reverse, depending on if 
"    " line contains std::cout"
"    let newstr = ""
"    if line =~ "std::cout"
"        let words = split(line," << ")
"        for word in words
"            " if token has these things then it's not a variable by itself
"            if word =~ "std::" || word =~ ": "
"                continue
"            endif
"            let newstr .= word . " "
"        endfor
"    else
"        let words = split(line)
"        let newstr .= "std::cout << "
"        for word in words
"            " if there's a quote in the variable, replace it with single tick
"            let escword = substitute(word, "\"", "'", "g")
"            let newstr .= " \" " . escword . ": \" << " . word . " << "
"        endfor
"        let newstr .= " std::endl;"
"    endif
"    :d
"    :-1put =newstr
"    execute "norm! =="
"endfu
"nnoremap <leader>cp :call g:CoutTokens()<CR>

map z/ <Plug>(incsearch-fuzzy-/)
map z? <Plug>(incsearch-fuzzy-?)
map zg/ <Plug>(incsearch-fuzzy-stay)
" Trying as a fix for mac
set backspace=indent,eol,start
autocmd FileType markdown match none
au BufNewFile,BufFilePre,BufRead *.markdown g/^$/d
au BufNewFile,BufFilePre,BufRead *.markdown imap <Space>  
au BufNewFile,BufFilePre,BufRead *.markdown let g:bufferline_echo=0
au BufNewFile,BufFilePre,BufRead *.markdown execute "normal G"

abbr ifnm if __name__ == "__main__":
abbr fui for (unsigned int i = 0; i < ; ++i)

" Related to note taking
nmap <leader>eo :rightbelow vnew<CR>:set cmdheight=4<CR>:e scp://uaf-10//home/users/phchang/public_html/jarvis/note.txt<CR>:set cmdheight=1<CR>
nmap <leader>er :set cmdheight=4<CR>:bufdo e<CR>:set cmdheight=1<CR>
nmap <leader>ew :set cmdheight=4<CR>:w<CR>:set cmdheight=1<CR>


" %!sort -V to sort by filename with number suffixes in correct order

set undodir=~/.vim/undoinfo/ " where to save undo histories
set undofile                " Save undos after file closes
set undolevels=100         " How many undos
set undoreload=100000        " number of lines to save for undo

nmap zz :tabnew %<CR>
nmap ZZ :wq<CR>

" vim-commentary
autocmd FileType c set commentstring=\/\/\ %s
autocmd FileType cuda set commentstring=\/\/\ %s
autocmd FileType cpp set commentstring=\/\/\ %s
autocmd FileType text set commentstring=\#\ %s
autocmd FileType crontab set commentstring=\#\ %s

nmap <C-k> :TagbarToggle<CR>

fu! GetBibItems()
    " Retrieve bibitems from inspire from arXiv number (the actual function is implemented in dot/mybashrc
    let line=getline('.')
    " turn into cout statement or reverse, depending on if 
    " line contains std::cout"
    let newstr = ""
    let cmd = "get_bibitem " . line
    let result = system(cmd)
    :d
    :-1put =result
endfu
nnoremap <leader>cb :call g:GetBibItems()<CR>

nnoremap <leader>vv :!/cvmfs/cms.cern.ch/external/tex/texlive/2017/bin/x86_64-linux/pdflatex skeleton.tex<CR><CR>

function! Mirror()
    try
        let v_save = @v
        normal! gv"vy
        let l = split(@v,'\n')
        call map(l,'join(reverse(split(v:val,"\\ze")),"")')
        call setreg('v',join(l,"\n"),visualmode())
        normal! $"vp
    finally
        let @v=v_save
    endtry
endfunction 
noremap  <silent> <leader>mr :<c-u>call Mirror()<cr>

fun! ShowFuncName()
  let lnum = line(".")
  let col = col(".")
  echohl ModeMsg
  echo getline(search("^[^ \t#/]\\{2}.*[^:]\s*$", 'bW'))
  echohl None
  call search("\\%" . lnum . "l" . "\\%" . col . "c")
endfun
map <leader>sf :call ShowFuncName() <CR>

let g:AutoPairsShortcutFastWrap = '<C-e>'

" nmap <C-i> :!./make<CR><CR>
" set wrap

set cindent cino=j1,(0,ws,Ws

nmap ]h <Plug>(GitGutterNextHunk)
nmap [h <Plug>(GitGutterPrevHunk)

nmap <Leader>w :w<CR>:!make<CR><CR>:!open main.pdf<CR><CR>
"eof

AC_INIT([BridgewellML])

VERSION=`head -n 1 CHANGELOG`
echo This is BridgewellML version $VERSION

# find R home and set CC/CFLAGS
: ${R_HOME=`R RHOME`}
if test -z "${R_HOME}"; then
  echo "could not determine R_HOME"
  exit 1
fi
RSCRIPT="${R_HOME}/bin/Rscript"

`"${RSCRIPT}" -e "library(Rcpp);compileAttributes('.')"`

AC_SUBST(VERSION)
AC_CONFIG_FILES(DESCRIPTION)
AC_OUTPUT

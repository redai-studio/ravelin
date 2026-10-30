Ravelin Documentation
=====================

AI application flows keep changing as models and products evolve. Once work
outlives a request, reconnects, later input, and recovery cross process and
replica boundaries.

Ravelin provides composable production mechanisms around an
application-defined flow. It keeps runtime resources portable, lets users
reconnect to live work while history stays application-owned, and makes selected
work and pending input survive worker replacement. The application or its
chosen orchestrator continues to own workflow and product semantics.

Ravelin's current execution integration is Haystack. See
:doc:`user-guide/why-ravelin` for the rationale and
:doc:`user-guide/comparison` for the decision boundary against integrated
frameworks, and
:doc:`user-guide/architecture-boundaries` for the precise compatibility and
ownership boundaries.

Start with :doc:`user-guide/why-ravelin` for the design direction and
:doc:`user-guide/architecture-boundaries` for the division of responsibility.
If your immediate concern is resource configuration, continue with
:doc:`user-guide/assets` before reading the complete
:doc:`user-guide/asset-resolution` contract.

.. toctree::
   :maxdepth: 2
   :caption: Guides

   user-guide/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   autoapi/index

.. toctree::
   :maxdepth: 1
   :caption: Project

   release_notes

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
